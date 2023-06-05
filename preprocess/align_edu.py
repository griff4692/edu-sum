import os
import regex as re

import argparse
import spacy
import numpy as np
from datasets import load_from_disk

from preprocess.convert_abstractive_to_extractive import _calc_rouge
from preprocess.convert_abstractive_to_extractive import gain_selection


def edus_from_html(text):
    edus = []
    tps = re.split(r'(</?e>)', text)
    for i, tp in enumerate(tps):
        if tp == '<e>':
            assert tps[i + 2] == '</e>'
            edus.append(tps[i + 1])
    return edus


def score_edu_pair(a, b):
    obj = _calc_rouge([[a]], [[b]])
    return (obj['rouge_1'] + obj['rouge_2']) / 2.0


def align_edus(sedus, tedus):
    scores = np.zeros([len(sedus), len(tedus)])
    for s_idx, sedu in enumerate(sedus):
        for t_idx, tedu in enumerate(tedus):
            score = score_edu_pair(sedu, tedu)
            scores[s_idx, t_idx] = score
    return scores


def align_example_edus(batch, nlp):
    oracle_align_idxs = []
    oracle_alignments = []
    oracle_soft_labels = []
    oracle_gain_idxs = []
    best_oracle_idxs = []
    gain_r1, gain_r2 = [], []
    best_r1, best_r2 = [], []
    gain_r1_v2, gain_r2_v2 = [], []
    align_r1, align_r2 = [], []
    for source_annot, target_annot, max_num in zip(
            batch['source_edu_annotated'], batch['target_edu_annotated'], batch['num_edus_post_trunc']
    ):
        source_edus = edus_from_html(source_annot)[:max_num]
        target_edus = edus_from_html(target_annot)

        score_matrix = align_edus(source_edus, target_edus)
        osl = list(map(float, np.max(score_matrix, axis=1).tolist()))
        oa = list(map(int, np.argmax(score_matrix, axis=0).tolist()))
        oi = list(sorted(list(set(oa))))

        source_edu_toks = [[x.text.strip() for x in nlp(edu) if len(x.text.strip()) > 0] for edu in source_edus]
        target_toks = [[x.text.strip() for x in nlp(edu) if len(x.text.strip()) > 0] for edu in target_edus]

        # Should 20 be higher for NYT?
        gain_idxs, gain_rouges, _, _, _ = gain_selection(
            source_edu_toks, target_toks, 20, lower=True, sort=True
        )

        align_oracle = [source_edus[i] for i in oi]
        gain_oracle = [source_edus[i] for i in gain_idxs]

        gain_obj = _calc_rouge([target_edus], [gain_oracle])
        align_obj = _calc_rouge([target_edus], [align_oracle])

        gain_r1.append(gain_obj['rouge_1'])
        gain_r2.append(gain_obj['rouge_2'])

        align_r1.append(align_obj['rouge_1'])
        align_r2.append(align_obj['rouge_2'])

        gain_r1_v2.append(gain_rouges['rouge_1'])
        gain_r2_v2.append(gain_rouges['rouge_2'])

        oracle_gain_idxs.append(gain_idxs)

        if gain_obj['rouge_1'] + gain_obj['rouge_2'] > align_obj['rouge_1'] + align_obj['rouge_2']:
            best_oracle_idxs.append(gain_idxs)
            best_r1.append(gain_obj['rouge_1'])
            best_r2.append(gain_obj['rouge_2'])
        else:
            best_oracle_idxs.append(oi)
            best_r1.append(align_obj['rouge_1'])
            best_r2.append(align_obj['rouge_2'])

        oracle_alignments.append(oa)
        oracle_align_idxs.append(oi)
        oracle_soft_labels.append(osl)

    return {
        'oracle_idxs': best_oracle_idxs,

        'oracle_alignments': oracle_alignments,
        'oracle_align_idxs': oracle_align_idxs,
        'oracle_gain_idxs': oracle_gain_idxs,
        'oracle_align_rouge1': align_r1,
        'oracle_align_rouge2': align_r2,
        'oracle_gain_rouge1': gain_r1,
        'oracle_gain_rouge2': gain_r2,
        'oracle_gain_rouge1_v2': gain_r1_v2,
        'oracle_gain_rouge2_v2': gain_r2_v2,
        'oracle_best_rouge1': best_r1,
        'oracle_best_rouge2': best_r2,
        'oracle_soft_labels': oracle_soft_labels
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Oracle Align Target EDUs to Source EDUs to the EDU-level datasets.')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--num_proc', default=64, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-use_pegasus', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.num_proc = 1

    pegasus_suffix = '_pegasus' if args.use_pegasus else ''

    nlp = spacy.load('en_core_web_sm')

    out_dir = os.path.join(args.data_dir, args.dataset + f'_edu_alignments{pegasus_suffix}')
    print(f'Saving to {out_dir}')

    print(f'Loading {args.dataset}...')
    edu_dir = os.path.join(args.data_dir, args.dataset + f'_edus{pegasus_suffix}')
    dataset = load_from_disk(edu_dir)

    metrics = [
        'oracle_align_rouge1', 'oracle_align_rouge2',
        'oracle_gain_rouge1', 'oracle_gain_rouge2',
        'oracle_gain_rouge1_v2', 'oracle_gain_rouge2_v2',
        'oracle_best_rouge1', 'oracle_best_rouge2'
    ]

    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        dataset[split] = dataset[split].map(
            lambda batch: align_example_edus(batch, nlp),
            batched=True, batch_size=1000, num_proc=args.num_proc,
        )

        print(f'{split} metrics...')
        for metric in metrics:
            val = str(round(np.mean(dataset[split][metric]), 3))
            print(f'{metric} -> {val}')
    if not args.debug:
        dataset.save_to_disk(out_dir)
