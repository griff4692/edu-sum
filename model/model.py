from itertools import combinations
from collections import defaultdict
import os
from copy import deepcopy
import regex as re

from datasets import load_metric
import pytorch_lightning as pl
from nltk import trigrams
import numpy as np
import spacy
from scipy.special import expit
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother
from transformers.models.bart.modeling_bart import BartForConditionalCopy

from preprocess.align_edu import edus_from_html
from preprocess.extract_oracles import convert_to_sents
from model.objectives import label_smoothed_unlikelihood
from eval.rouge_metric import RougeMetric
from eval.diversity import diversity_score


def edu_reps(mask, h):
    assert mask.sum() % 2 == 1  # Assert Odd

    edu_idxs = torch.where(mask)[0]

    assert int(edu_idxs[0].item()) == 0
    cls_h = [h[edu_idxs[0]]]
    for start in range(1, len(edu_idxs), 2):
        pooled = h[edu_idxs[start]:edu_idxs[start + 1] + 1].mean(dim=0)
        cls_h.append(pooled)

    return torch.stack(cls_h).unsqueeze(0)


class TransformerSummarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_model):
        super().__init__()

        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
        self.config = self.model.config
        self.model.resize_token_embeddings(len(tokenizer))
        self.lr = self.hparams.lr  # Necessary for tune_lr to work with PytorchLightning
        self.rouge = load_metric('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)
        self.nlp = spacy.load('en_core_web_sm')
        self.rouge_metric = RougeMetric()

        self.sent_bart = None
        if 'extract' in self.hparams.summary_style:
            self.sent_config = deepcopy(self.config)
            # Should be tunable
            self.sent_config.encoder_layers = 2
            self.sent_config.decoder_layers = 2
            self.sent_config.classifier_dropout = self.hparams.copy_bart_class_dropout
            self.sent_config.forced_bos_token_id = None
            self.sent_config.forced_eos_token_id = None
            # (everything else is copied from other BARTEncoder)
            # <s> is never used but there as padding since it's id is 0
            self.sent_config.vocab_size = 3  # <s> <pad> </s>
            self.sent_bart = BartForConditionalCopy(self.sent_config)
            self.stop_embed = nn.Embedding(
                num_embeddings=1, embedding_dim=self.sent_config.d_model, padding_idx=None
                )

    def shared_step(self, batch, source=None, build_extracts=True):
        metrics = {}
        extracts = None
        return_loss = 0

        encoder_inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        encoder_outputs = self.get_encoder_h(encoder_inputs)
        encoder_h = encoder_outputs.last_hidden_state

        if 'extract' in self.hparams.summary_style:
            # Generate Sentence Plan with separate randomly initialized Bart Decoder (self.sent_bart)
            extract_loss, salience_loss, extracts = self.generate_extracts(batch, source, encoder_h)
            metrics['extract'] = extract_loss
            return_loss += self.hparams.mle_weight * extract_loss
            if salience_loss is not None:
                return_loss += self.hparams.salience_weight * salience_loss
                metrics['salience'] = salience_loss

        # score is just extraction (no word-level generation)
        plan_encoder_outputs = None
        if 'abstract' in self.hparams.summary_style:
            updated_inputs = {
                'encoder_outputs': encoder_outputs,
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
            }

            output = self.model(**updated_inputs, use_cache=False)
            # Regular MLE decoder loss
            metrics['loss'] = output.loss  # Log unsmoothed loss for comparison to earlier runs.
            # Return label-smoothed loss for BART Decoder
            smooth_lm_loss = self.label_smoother(output, batch['labels'])

            if self.hparams.extract_indicators:
                plan_encoder_inputs = {
                    'input_ids': batch['plan_input_ids'], 'attention_mask': batch['plan_attention_mask']
                }
                plan_encoder_outputs = self.get_encoder_h(plan_encoder_inputs)
                updated_inputs = {
                    'encoder_outputs': plan_encoder_outputs,
                    'attention_mask': batch['plan_attention_mask'],
                    'labels': batch['labels'],
                }

                output = self.model(**updated_inputs, use_cache=False)
                like_smooth = self.label_smoother(output, batch['labels'])
                metrics['likelihood'] = like_smooth
                return_loss += self.hparams.like_coef * like_smooth

                corrupt_encoder_inputs = {
                    'input_ids': batch['corrupt_input_ids'], 'attention_mask': batch['corrupt_attention_mask']
                }
                corrupt_encoder_outputs = self.get_encoder_h(corrupt_encoder_inputs)
                updated_inputs = {
                    'encoder_outputs': corrupt_encoder_outputs,
                    'attention_mask': batch['corrupt_attention_mask'],
                    'labels': batch['labels'],
                }

                output = self.model(**updated_inputs, use_cache=False)
                # Add unlikelihood loss
                probs_neg = torch.softmax(output.logits, dim=-1)
                unlike_smooth, _ = label_smoothed_unlikelihood(probs_neg, batch['labels'], reduce=True)
                metrics['unlikelihood'] = unlike_smooth
                return_loss += self.hparams.unlike_coef * unlike_smooth

            return_loss += self.hparams.mle_weight * smooth_lm_loss
        return {
            'metrics': metrics, 'return_loss': return_loss, 'encoder_outputs': encoder_outputs, 'extracts': extracts,
            'plan_encoder_outputs': plan_encoder_outputs,
        }

    def training_step(self, batch, batch_idx):
        if self.hparams.summary_style == 'extract':
            batch['cls_mask'][:, 0] = True  # Document <s> Token gets passed to the sentence encoder
        # source = self.parse_source_text_from_inputs(batch)
        shared_output = self.shared_step(batch, source=None, build_extracts=False)  # Don't generate extracts
        metrics, return_loss = shared_output['metrics'], shared_output['return_loss']
        metrics['combined'] = return_loss
        self.log_metrics(metrics, is_train=True)
        return return_loss

    def score_extracts(self, batch, source, encoder_h, build=True):
        batch_size = len(batch['cls_mask'])
        losses = []
        summaries = [] if build else None
        for batch_idx in range(batch_size):
            cls_h = encoder_h[batch_idx, batch['cls_mask'][batch_idx], :]
            labels = batch['oracle_labels'][batch_idx]
            sent_preds = self.sent_classifier(cls_h).squeeze(-1)
            labels_onehot = torch.zeros_like(sent_preds).to(self.device)
            labels_onehot[labels] = 1
            loss = self.sent_loss(sent_preds, labels_onehot)
            losses.append(loss)
            if summaries is not None:
                y_hat_np = sent_preds.flatten().detach().cpu().numpy()
                sum = self.build_summaries(source=source[batch_idx], y_hat=y_hat_np)
                summaries.append(sum)
        losses = torch.stack(losses).mean()
        return losses, summaries

    def validation_step(self, batch, batch_idx):
        if self.hparams.summary_style == 'extract':
            batch['cls_mask'][:, 0] = True  # Document <s> Token gets passed to the sentence encoder
        batch_size = len(batch['input_ids'])

        source = self.parse_source_text_from_inputs(batch)
        shared_output = self.shared_step(batch, source=source, build_extracts=True)
        metrics, return_loss, extracts = shared_output['metrics'], shared_output['return_loss'], shared_output['extracts']
        metrics['combined'] = return_loss
        self.log_metrics(metrics, is_train=False)
        extract_outputs = [None for _ in range(batch_size)]
        if extracts is not None:
            for batch_idx in range(batch_size):
                extract_outputs[batch_idx] = {
                    'source': source[batch_idx],
                    'abstracts': None,
                    'extracts': extracts[batch_idx],
                    'reference': batch['references'][batch_idx],
                }

        gen_outputs = [None for _ in range(batch_size)]
        validation_kwargs = {
            'num_beams': 1,
            'num_return_sequences': 1,  # Don't over-generate for validation
            'references': batch['references'],
        }
        if self.hparams.summary_style != 'extract':
            gen_outputs = self.shared_generate(
                batch, source, **validation_kwargs, encoder_outputs=shared_output['encoder_outputs'],
                plan_encoder_outputs=shared_output['plan_encoder_outputs']
            )

        # Merge the two if needed (score_abstract only)
        outputs_resolved = self.merge_outputs(gen_outputs, extract_outputs)

        # It's a list of dictionaries --> convert into dictionary of lists and process as a batch (for ROUGE)
        output_dict = defaultdict(list)
        for item in outputs_resolved:
            for k, v in item.items():
                if type(v) == list:
                    output_dict[k] += v
                elif v is not None:
                    output_dict[k].append(v)
        # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
        eval_metrics = {}
        if len(output_dict['abstracts']) > 0:
            eval_metrics.update(self.compute_rouge(output_dict['abstracts'], batch['references']))

        if len(output_dict['extracts']) > 0:
            extracts = [x['summary'] for x in output_dict['extracts']]
            eval_metrics.update(self.compute_rouge(extracts, batch['references'], prefix='extract_'))

        self.log_metrics(eval_metrics, is_train=False)
        return return_loss

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        if self.hparams.summary_style == 'extract':
            batch['cls_mask'][:, 0] = True  # Document <s> Token gets passed to the sentence encoder
        source = self.parse_source_text_from_inputs(batch)
        use_hf_rouge = gen_kwargs.pop('use_hf_rouge')
        eval = not use_hf_rouge
        references = batch['references']
        batch_size = len(references)
        extract_outputs = [None for _ in range(batch_size)]
        if 'extract' in self.hparams.summary_style:
            # Predict if a sentence is in oracle summary
            encoder_kwargs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            output = self.model.model.encoder(**encoder_kwargs)
            encoder_h = output.last_hidden_state

            extractive_summaries, _ = self.sample_gen_extracts(
                batch, source, encoder_h, **gen_kwargs
            )

            for batch_idx in range(batch_size):
                extract_outputs[batch_idx] = {
                    'source': source[batch_idx],
                    'abstracts': None,
                    'extracts': extractive_summaries[batch_idx],
                    'reference': references[batch_idx],
                }

        gen_outputs = [None for _ in range(batch_size)]
        gen_kwargs.update({
            'references': references
        })
        if self.hparams.summary_style != 'extract':
            gen_outputs = self.shared_generate(batch, source, **gen_kwargs)

        # Merge the two if needed (score_abstract only)
        outputs_resolved = self.merge_outputs(gen_outputs, extract_outputs)

        batch_outputs = []
        for batch_idx, (reference, gen_output) in enumerate(zip(references, outputs_resolved)):
            abstract_flat = '' if gen_output['abstracts'] is None else '<cand>'.join(gen_output['abstracts'])
            extract_flat = '' if gen_output['extracts'] is None else '<cand>'.join(
                [x['summary'] for x in gen_output['extracts']])
            extract_idx_flat = '' if gen_output['extracts'] is None else '<cand>'.join(
                [','.join(map(str, x['idxs'])) for x in gen_output['extracts']])
            save_out = {
                'abstract': abstract_flat, 'extract': extract_flat,
                'reference': reference, 'source': gen_output['source']['text'],
                'extract_idx': extract_idx_flat,
            }

            if (
                    gen_output['extracts'] is not None
                    and len(gen_output['extracts']) > 0
                    and 'sent_dist' in gen_output['extracts'][0]
            ):
                dist_flat = '<cand>'.join([','.join(list(map(str, x['sent_dist']))) for x in gen_output['extracts']])
                save_out['sent_scores'] = dist_flat

            if (
                    gen_output['extracts'] is not None
                    and len(gen_output['extracts']) > 0
                    and 'beam_score' in gen_output['extracts'][0]
            ):
                dist_flat = ','.join([str(x['beam_score']) for x in gen_output['extracts']])
                save_out['extract_beam_scores'] = dist_flat

            # If we just generate a plan there is only an "extracted" (from plan) summary.  No generation
            if gen_output['abstracts'] is not None:  # Take top of the beam or first returned sequence
                save_out.update(self.compute_rouge(gen_output['abstracts'][:1], [reference], eval=eval))
                if len(gen_output['abstracts']) > 1:
                    # Get all the ROUGE abstracts (average ROUGE-1, ROUGE-2)
                    abstract_cand_metrics, best_abstract_metric, avg_abstract_r1, diversity = self.score_candidates(
                        [reference], gen_output['abstracts'], 'abstract', eval=eval
                    )
                    save_out.update(best_abstract_metric)
                    save_out.update({'avg_rouge1_f1': avg_abstract_r1})
                    save_out.update({'diversity': diversity})
                    save_out['abstract_rouges'] = ','.join(
                        [str(x['best_abstract_mean_f1']) for x in abstract_cand_metrics]
                    )

            if gen_output['extracts'] is not None:
                extracts = [x['summary'] for x in gen_output['extracts']]
                try:
                    save_out.update(self.compute_rouge(extracts[:1], [reference], prefix='extract_', eval=eval))
                except:
                    print(f'Could not compute ROUGE score for {extracts[:1]}')
                    print('Reference is ', reference)
                if len(extracts) > 1:
                    extract_cand_metrics, best_extract_metric, avg_r1, extract_diversity = self.score_candidates(
                        [reference], extracts, 'extract', eval=eval
                    )

                    save_out.update({'avg_extract_rouge1_f1': avg_r1})
                    save_out.update({'extract_diversity': extract_diversity})
                    save_out.update(best_extract_metric)
                    save_out['extract_rouges'] = ','.join(
                        [str(x['best_extract_mean_f1']) for x in extract_cand_metrics]
                    )

            batch_outputs.append(save_out)

        return batch_outputs

    def get_encoder_h(self, batch):
        return self.model.model.encoder(**batch)

    def get_eos(self, seq_len):
        return seq_len - 1  # Decrement for Document Token in first position

    def compute_gen_extract_loss(self, cls_mask, encoder_h, oracle_labels, oracle_soft_labels):
        kld_loss = nn.KLDivLoss(reduction='none')

        batch_size = len(cls_mask)
        losses = []
        soft_losses = []
        sent_encoder_h = []
        stop_input_id = torch.LongTensor([0]).to(self.device)
        for batch_idx in range(batch_size):
            row_h = encoder_h[batch_idx]
            mask = cls_mask[batch_idx]

            cls_h = edu_reps(mask, row_h)
            eos_id = self.get_eos(cls_h.size()[1])
            labels = oracle_labels[batch_idx]
            soft_labels = oracle_soft_labels[batch_idx]
            eos_dummy = torch.LongTensor([eos_id]).to(self.device)
            labels = torch.cat([labels, eos_dummy]).unsqueeze(0)
            # Concatenate
            inputs_embeds = torch.cat([cls_h, self.stop_embed(stop_input_id).unsqueeze(0)], dim=1)
            output = self.sent_bart(inputs_embeds=inputs_embeds, labels=labels, output_hidden_states=True)
            # loss = self.label_smoother(output, labels)
            loss = output.loss
            sent_encoder_h.append(output.encoder_last_hidden_state)

            # Don't include the document token for the salience classifier (first position)
            # Also don't include the dummy STOP token at position -1
            # [DOC, EDU 1, EDU 2, ... EDU n, STOP]
            sal_scores = self.sent_bart.salience_classifier(output.encoder_last_hidden_state[0, 1:-1]).squeeze(-1)
            target_dist = torch.softmax(soft_labels.half() * self.hparams.salience_temp, dim=0)
            kl_loss = kld_loss(torch.log_softmax(sal_scores, dim=0), target_dist).sum()
            soft_losses.append(kl_loss)

            losses.append(loss)
        avg_losses = torch.stack(losses).mean()
        avg_soft_losses = torch.stack(soft_losses).mean()
        return avg_losses, avg_soft_losses, sent_encoder_h

    def sample_score_extracts(self, batch, source, encoder_h, num_return_sequences=1, topk=10, joint_rank=True):
        batch_size = len(batch['cls_mask'])
        losses = []
        summaries = []
        for batch_idx in range(batch_size):
            cls_h = encoder_h[batch_idx, batch['cls_mask'][batch_idx], :]
            labels = batch['oracle_labels'][batch_idx]
            sent_preds = self.sent_classifier(cls_h).squeeze(-1)
            labels_onehot = torch.zeros_like(sent_preds).to(self.device)
            labels_onehot[labels] = 1
            loss = self.sent_loss(sent_preds, labels_onehot)
            losses.append(loss)
            y_hat = sent_preds.flatten().detach().cpu()
            y_hat_np = y_hat.numpy()

            if num_return_sequences == 1:
                sum = [self.build_summaries(source=source[batch_idx], y_hat=y_hat_np)]
            else:
                sum = []
                sample_num = num_return_sequences
                k = min(topk, len(y_hat_np))
                top_k_y = y_hat.topk(k=k)
                top_k_y_indices = top_k_y.indices
                temperature = 1.

                top_k_y_p = torch.softmax(top_k_y.values * temperature, dim=0).numpy()
                if joint_rank:
                    all_combos = list(combinations(np.arange(k), 3))
                    triplet_scores = []
                    for ic in all_combos:
                        a, b, c = ic
                        triplet_scores.append(
                            top_k_y_p[a] * top_k_y_p[b] * top_k_y_p[c]
                        )
                    triplet_idxs = np.argsort(-np.array(triplet_scores))[:sample_num]
                    selected_ranks = [all_combos[i] for i in triplet_idxs]
                    selected_idxs = [
                        [top_k_y_indices[i].item() for i in ir] for ir in selected_ranks
                    ]
                else:
                    selected_idxs = []
                    for sample in range(sample_num):
                        try:
                            summary_idx = list(np.random.choice(top_k_y_indices, size=(3,), replace=False, p=top_k_y_p))
                        except:
                            print(top_k_y_indices)
                            summary_idx = list(np.random.choice(top_k_y_indices, size=(3,), replace=False))
                        selected_idxs.append(summary_idx)
                for summary_idx in selected_idxs:
                    return_obj = self.get_summary_from_sent_idxs(source[batch_idx], summary_idx)
                    return_obj['sent_dist'] = y_hat_np
                    sum.append(return_obj)
            # If we are sampling, get num_return_sequences samples of size 3 from top K predictions
            summaries.append(sum)
        return summaries

    def generate_with_sent_bart(self, **kwargs):
        with torch.no_grad():
            self.sent_bart.prepare_counter_for_generation(eos_token_id=kwargs['eos_token_id'])
            pred_ids = self.sent_bart.generate(**kwargs)
            self.sent_bart.reset_counter_for_generation()
            return pred_ids

    def sample_gen_extracts(self, batch, source, encoder_h, **gen_kwargs):
        extractive_summaries = []
        raw_predictions = []
        cls_mask = batch['cls_mask']
        batch_size = len(cls_mask)
        stop_input_id = torch.LongTensor([0]).to(self.device)
        for batch_idx in range(batch_size):
            cls_h = edu_reps(cls_mask[batch_idx], encoder_h[batch_idx])
            inputs_embeds = torch.cat([cls_h, self.stop_embed(stop_input_id).unsqueeze(0)], dim=1)
            eos_token_id = self.get_eos(cls_h.size()[1])
            fixed_kwargs = {
                'inputs_embeds': inputs_embeds,
                'eos_token_id': eos_token_id,
                'early_stopping': True,
                'output_scores': True,
                'return_dict_in_generate': True,
            }

            fixed_kwargs.update(**gen_kwargs)
            outputs = self.generate_with_sent_bart(**fixed_kwargs)
            beam_scores = outputs.sequences_scores.cpu().numpy().tolist()

            pred_ids = outputs.sequences

            sent_labels = [
                list(set(self.remove_special_tokens_from_sent_bart(pred_id.cpu().numpy().tolist(), eos_token_id)))
                for pred_id in pred_ids
            ]

            sent_encoder_h = self.sent_bart.model.encoder(inputs_embeds=inputs_embeds).last_hidden_state
            encoder_sent = sent_encoder_h[:, 1:, :]
            assert len(encoder_sent) == 1
            pooled_extract = torch.stack([encoder_sent[0, x].mean(dim=0) for i, x in enumerate(sent_labels)])
            encoder_doc_rep = sent_encoder_h[:, 0, :].repeat(len(sent_labels), 1)

            raw_predictions.append(pred_ids)
            return_obj = []
            for pred_idx, summary_idx in enumerate(pred_ids.tolist()):
                summary_idx_no_special = self.remove_special_tokens_from_sent_bart(summary_idx, eos_token_id)
                summary_idx_no_special_no_dup = []
                seen = set()
                for sidx in summary_idx_no_special:
                    if sidx in seen:
                        print(f'Duplicated generated sentence {sidx}. Removing.')
                        continue
                    summary_idx_no_special_no_dup.append(sidx)
                    seen.add(sidx)
                summary_obj = self.get_summary_from_sent_idxs(source[batch_idx], summary_idx_no_special_no_dup)
                summary_obj['beam_score'] = beam_scores[pred_idx]
                return_obj.append(summary_obj)
            extractive_summaries.append(return_obj)
        return extractive_summaries, raw_predictions

    def generate_extracts(self, batch, source, encoder_h):
        cls_mask = batch['cls_mask']
        loss, salience_loss, sent_encoder_h = self.compute_gen_extract_loss(
            cls_mask, encoder_h, batch['oracle_labels'], batch['oracle_soft_labels'],
        )
        summaries = None
        if not self.sent_bart.training:
            gen_kwargs = {
                'min_length': 3,  # 2 without the special token
                'max_length': 20,
                'num_beams': 4,
            }
            summaries, _ = self.sample_gen_extracts(batch, source, encoder_h, **gen_kwargs)

        return loss, salience_loss, summaries

    def build_summaries(self, source, y_hat, trigram_block=True, max_num_sents=3):
        all_summaries = []
        priority = expit(y_hat.copy())
        trigram_to_sent_idx = sent_idx_to_trigram = None
        if trigram_block:
            trigram_to_sent_idx = defaultdict(list)
            sent_idx_to_trigram = defaultdict(list)

            for sent_idx, sent in enumerate(source['sents']):
                sent_toks = [t for t in re.sub('\W+', ' ', sent.lower()).split(' ') if len(t) > 0]
                sent_trigrams = list(trigrams(sent_toks))
                for trigram in sent_trigrams:
                    trigram_to_sent_idx[trigram].append(sent_idx)
                    sent_idx_to_trigram[sent_idx].append(trigram)

        for k in range(min(max_num_sents, len(source['sents']))):
            top_sent = np.argmax(priority)
            priority[top_sent] = float('-inf')
            if trigram_block:
                for trigram in sent_idx_to_trigram[top_sent]:
                    for other_sent_idx in trigram_to_sent_idx[trigram]:
                        priority[other_sent_idx] = float('-inf')
            # Matching Trigrams
            prev_sents = [] if k == 0 else all_summaries[k - 1]
            summary_at_k = prev_sents + [top_sent]
            all_summaries.append(summary_at_k)
        summary_idx = all_summaries[-1]
        return_obj = self.get_summary_from_sent_idxs(source, summary_idx)
        return_obj['sent_dist'] = y_hat
        return return_obj

    def shared_generate(self, batch, source, references, encoder_outputs=None, plan_encoder_outputs=None, **gen_kwargs):
        fixed_kwargs = {  # Some of these values may get overridden by gen_kwargs
            'attention_mask': batch['attention_mask'],
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'output_scores': True
        }
        if plan_encoder_outputs is not None:
            fixed_kwargs['attention_mask'] = batch['plan_attention_mask']
            fixed_kwargs['encoder_outputs'] = plan_encoder_outputs
        elif encoder_outputs is not None:
            fixed_kwargs['encoder_outputs'] = encoder_outputs
        else:
            fixed_kwargs['input_ids'] = batch['input_ids']

        # Update them with user-specific kwargs
        fixed_kwargs.update(gen_kwargs)
        pred_ids = self.model.generate(**fixed_kwargs)
        gold_ids = batch['labels']
        gold_ids[torch.where(batch['labels'] == -100)] = 1

        batch_size = len(batch['input_ids'])
        num_pred = len(pred_ids)
        num_cands = gen_kwargs['num_return_sequences']
        assert num_cands * batch_size == num_pred
        pred_ids = pred_ids.view(batch_size, num_cands, -1) if num_cands > 1 else pred_ids.unsqueeze(1)
        return [
            self.parse_output(source[batch_idx], references[batch_idx], pred_ids[batch_idx])
            for batch_idx in range(batch_size)
        ]

    def get_summary_from_sent_idxs(self, source, extractive_idxs, sort=False):
        if sort:
            extractive_idxs = list(sorted(extractive_idxs))
        summary = ' '.join([source['sents'][i].strip() for i in extractive_idxs])
        return {
            'idxs': extractive_idxs,
            'summary': summary
        }

    def parse_output(self, source, reference, pred_ids):
        pred_str = self.tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        num_pred = len(pred_str)
        if self.hparams.summary_style == 'extract_abstract':
            abstracts = pred_str
            extracts = None  # these will be filled in after (not part of generation)
        elif self.hparams.summary_style == 'abstract':
            extracts = None
            abstracts = pred_str
        else:
            raise Exception(f'Unrecognized summary style -> {self.hparams.summary_style}')

        return {
            'source': source,
            'reference': reference,
            'abstracts': abstracts,
            'extracts': extracts,
        }

    def remove_special_tokens_from_sent_bart(self, summary_idx, dynamic_eos_token_id):
        assert summary_idx[0] == self.sent_config.decoder_start_token_id
        end_idx = summary_idx.index(dynamic_eos_token_id)
        trunc_idx = np.array(summary_idx[end_idx + 1:])
        if len(trunc_idx) > 0 and np.all(trunc_idx != self.sent_bart.config.pad_token_id):
            trunc_str = ','.join([str(x) for x in summary_idx[1:end_idx]])
            full_str = ','.join([str(x) for x in summary_idx])
            print(f'Warning! Truncating non-padding tokens: {full_str} -> {trunc_str}')
        return summary_idx[1: end_idx]

    def compute_rouge(self, generated, gold, prefix='', eval=False, rouge_types=['rouge1', 'rouge2', 'rougeL']):
        for i in range(len(generated)):
            if len(generated[i]) == 0:
                generated[i] = 'empty'
                print('Warning: Empty generated sequence. Setting to string empty for ROUGE scoring.')
            if len(gold[i]) == 0:
                gold[i] = 'empty'
                print('Warning: Empty reference sequence. Setting to string empty for ROUGE scoring.')

        if eval:  # Use SummEval PERL script
            outputs = self.rouge_metric.evaluate_batch(generated, gold, aggregate=True)['rouge']
            f1s = []
            stats = {}
            for rouge_type in ['1', '2', 'L']:
                fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
                stats[f'{prefix}rouge{rouge_type}_precision'] = outputs[f'rouge_{rouge_type.lower()}_precision']
                stats[f'{prefix}rouge{rouge_type}_recall'] = outputs[f'rouge_{rouge_type.lower()}_recall']
                stats[f'{prefix}rouge{rouge_type}_f1'] = fscore
                f1s.append(fscore)
        else:
            rouge_output = self.rouge.compute(predictions=generated, references=gold, rouge_types=rouge_types)
            stats = {}
            f1s = []
            for rouge_type in rouge_types:
                stats[f'{prefix}{rouge_type}_precision'] = rouge_output[rouge_type].mid.precision
                stats[f'{prefix}{rouge_type}_recall'] = rouge_output[rouge_type].mid.recall
                stats[f'{prefix}{rouge_type}_f1'] = rouge_output[rouge_type].mid.fmeasure
                f1s.append(rouge_output[rouge_type].mid.fmeasure)
        stats[f'{prefix}mean_f1'] = np.array(f1s).mean()
        return stats

    def parse_source_text_from_inputs(self, batch):
        source_special = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        source_no_special = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=True)
        if self.hparams.add_sent_toks:  # If we have sentences demarcated in the source text
            source_sents = [edus_from_html(x) for x in source_special]
            source_sent_toks = [
                [[str(t.text).strip() for t in self.nlp(edu) if len(str(t.text).strip()) > 0] for edu in edus]
                for edus in source_sents
            ]
        else:  # We have to sentence split ourselves
            source_sents = [
                convert_to_sents(x, self.nlp, is_dialogue=self.hparams.dataset == 'samsum') for x in source_no_special
            ]
            source_sent_toks = [
                [[str(t.text).strip() for t in sentence if len(str(t.text).strip()) > 0] for sentence in doc]
                for doc in source_sents
            ]
            source_sents = list(map(lambda ss: [str(x).strip() for x in ss], source_sents))
        return [{'text': a.strip(), 'sents': b, 'sent_toks': c} for a, b, c in
                zip(source_no_special, source_sents, source_sent_toks)]

    def merge_outputs(self, gen_outputs, score_outputs):
        if self.hparams.summary_style == 'extract':
            return score_outputs
        merged = []
        assert len(score_outputs) == len(gen_outputs)
        for batch_idx in range(len(score_outputs)):
            row = {}
            if gen_outputs[batch_idx] is not None:
                row.update(gen_outputs[batch_idx])
            if score_outputs[batch_idx] is not None:
                for k, v in score_outputs[batch_idx].items():
                    if v is None or row[k] is not None:
                        continue
                    row[k] = v
            merged.append(row)
        return merged

    def score_candidates(self, reference, candidates, prefix, eval=False):
        cand_metrics = [self.compute_rouge(
            [abstract], reference, prefix=f'best_{prefix}_', eval=False, rouge_types=['rouge1']
        ) for abstract in candidates]
        cand_scores = [x[f'best_{prefix}_mean_f1'] for x in cand_metrics]
        avg_r1_f1 = np.mean([x[f'best_{prefix}_rouge1_f1'] for x in cand_metrics])
        best_cand = np.argmax(cand_scores)
        best_metric = self.compute_rouge([candidates[best_cand]], reference, prefix=f'best_{prefix}_', eval=eval)
        diversity = 0
        if len(candidates) >= 2:
            try:
                diversity = np.mean(diversity_score(candidates))
            except IndexError:
                print('Index error when using NLTK for diversity score. Setting diversity to None.')
                diversity = None
        return cand_metrics, best_metric, avg_r1_f1, diversity

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        nps = list(self.named_parameters())

        grouped_parameters = [
            {
                'params': [p for n, p in nps if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay,
                'lr': self.lr,
            },
            {
                'params': [p for n, p in nps if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': self.lr,
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)
        if self.hparams.no_schedule or self.hparams.debug or self.hparams.find_lr:
            return optimizer

        # 6% is somewhat standard for fine-tuning Transformers (can be a tunable hyper-parameter as well)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps
        )

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items

    def log_metrics(self, metrics, is_train):
        for k, v in metrics.items():
            split_str = 'train' if is_train else 'validation'
            self.log(f'{split_str}/{k}', v, on_epoch=not is_train, on_step=is_train, prog_bar=True, sync_dist=True)
