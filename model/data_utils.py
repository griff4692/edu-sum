import os
from pathlib import Path

import nltk
import torch
import numpy as np


def postprocess_text(texts):
    return ['\n'.join(nltk.sent_tokenize(text.strip())) for text in texts]


def infer_dataset(args, col):
    if args.dataset is None:
        rel_name = getattr(args, col)
        if 'samsum' in rel_name:
            args.dataset = 'samsum'
        elif 'nyt' in rel_name:
            args.dataset = 'nyt'
        elif 'cnn' in rel_name:
            args.dataset = 'cnn_dailymail'
        elif 'xsum' in rel_name:
            args.dataset = 'xsum'
        else:
            raise Exception(f'Cant infer dataset from {rel_name}. Must pass with --dataset flag.')


class Seq2SeqCollate:
    def __init__(
            self, tokenizer, max_input_length=1024, split=None, verbose=False,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.split = split
        additional_ids = self.tokenizer.additional_special_tokens_ids
        self.special_id_min = 999999 if len(additional_ids) == 0 else min(self.tokenizer.additional_special_tokens_ids)
        self.verbose = verbose

    def __call__(self, batch_list):
        # Pad input_ids
        labels = [x['labels'] for x in batch_list]

        input_ids = [x['input_ids'] for x in batch_list]
        input_seq_lens = [len(x) for x in input_ids]
        max_input_len = max(input_seq_lens)

        input_ids_pad = [
            x + [self.tokenizer.pad_token_id] * (max_input_len - input_seq_lens[i])
            for i, x in enumerate(input_ids)
        ]
        input_ids_pad = torch.from_numpy(np.array(input_ids_pad, dtype=np.int64))
        attention_mask = (input_ids_pad != self.tokenizer.pad_token_id).float()

        label_seq_lens = [len(x) for x in labels]
        max_label_len = max(label_seq_lens)
        label_ids_pad = [
            x + [-100] * (max_label_len - label_seq_lens[i])
            for i, x in enumerate(labels)
        ]
        label_ids_pad = torch.from_numpy(np.array(label_ids_pad, dtype=np.int64))

        oracle_labels = [
            torch.LongTensor(x['oracle_labels']) if x['oracle_labels'] is not None else None for x in batch_list
        ]

        oracle_soft_labels = [
            torch.FloatTensor(x['oracle_soft_labels']) if x['oracle_soft_labels'] is not None else None
            for x in batch_list
        ]

        references = [x['reference'] for x in batch_list]
        row = {
            'input_ids': input_ids_pad,
            'attention_mask': attention_mask,
            'labels': label_ids_pad,
            'cls_mask': input_ids_pad >= self.special_id_min,
            'oracle_labels': oracle_labels,
            'oracle_soft_labels': oracle_soft_labels,
            'references': references,
        }

        if 'corrupt_input_ids' in batch_list[0]:
            corrupt_input_ids = [x['corrupt_input_ids'] for x in batch_list]
            corrupt_input_seq_lens = [len(x) for x in corrupt_input_ids]
            corrupt_input_ids_pad = torch.from_numpy(np.array([
                x + [self.tokenizer.pad_token_id] * (max(corrupt_input_seq_lens) - corrupt_input_seq_lens[i])
                for i, x in enumerate(corrupt_input_ids)
            ], dtype=np.int64))
            corrupt_attention_mask = (corrupt_input_ids_pad != self.tokenizer.pad_token_id).float()
            row['corrupt_input_ids'] = corrupt_input_ids_pad
            row['corrupt_attention_mask'] = corrupt_attention_mask

        if 'plan_input_ids' in batch_list[0]:
            plan_input_ids = [x['plan_input_ids'] for x in batch_list]
            plan_input_seq_lens = [len(x) for x in plan_input_ids]
            plan_input_ids_pad = torch.from_numpy(np.array([
                x + [self.tokenizer.pad_token_id] * (max(plan_input_seq_lens) - plan_input_seq_lens[i])
                for i, x in enumerate(plan_input_ids)
            ], dtype=np.int64))
            plan_attention_mask = (plan_input_ids_pad != self.tokenizer.pad_token_id).float()
            row['plan_input_ids'] = plan_input_ids_pad
            row['plan_attention_mask'] = plan_attention_mask

        return row


def get_path_from_exp(weights_dir, experiment=None):
    if experiment is None:
        dir = weights_dir
    else:
        dir = os.path.join(weights_dir, experiment)
    paths = list(Path(dir).rglob('*.ckpt'))
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')
