import os
import numpy as np
import regex as re

from tqdm import tqdm
from datasets import load_from_disk
import pandas as pd


def _tokenize(text):
    return [x.strip() for x in re.split(r'(\W+)', text.lower()) if len(x.strip()) > 0]


def find_idx(full, arr, toks):
    overlaps = []
    for x in arr:
        o = len(set(x).intersection(set(toks)))
        v = o / (0.5 * len(set(x)) + 0.5 * len(set(toks)))
        overlaps.append(v)
    max_arrs = [(f, x) for val, f, x in zip(overlaps, full, arr) if val == max(overlaps)]
    length_diffs = []
    for arr in max_arrs:
        length_diffs.append(abs(len(arr[1]) - len(toks)))
    return max_arrs[int(np.argmin(length_diffs))][0], len(max_arrs) - 1


def resolve(data_row, cand_fn, ref_fn, preds):
    with open(cand_fn, 'r') as fd:
        brio_pred = fd.read()
    with open(ref_fn, 'r') as fd:
        brio_ref = fd.read()

    ref_toks = _tokenize(brio_ref)
    highlight_toks = _tokenize(data_row['highlights'])
    o = len(set(ref_toks).intersection(set(highlight_toks)))
    v = o / (0.5 * len(set(ref_toks)) + 0.5 * len(set(highlight_toks)))
    assert v >= 0.8

    brio_pred_toks = _tokenize(brio_pred)
    pred_toks = [_tokenize(x) for x in preds]
    aligned, potential_conflicts = find_idx(preds, pred_toks, brio_pred_toks)

    return aligned, brio_pred, potential_conflicts


if __name__ == '__main__':
    test = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')['test']
    n = len(test)

    ea_preds = pd.read_csv('/nlp/projects/faithsum/results/cnn_e_v1/test_from_beam_16_extract_cnn_ea_rand_v2.csv')
    assert ea_preds['dataset_idx'].tolist() == list(range(n))
    ea_cands = os.path.expanduser('~/BRIO/result/cnn_e_v1_ea_rand_v2/candidate_ranking')
    ea_refs = os.path.expanduser('~/BRIO/result/cnn_e_v1_ea_rand_v2/reference_ranking')

    dbs_preds = pd.read_csv('/nlp/projects/faithsum/results/bart_large_cnn/test_diverse_16_outputs.csv')
    assert ea_preds['dataset_idx'].tolist() == list(range(n))
    dbs_cands = os.path.expanduser('~/BRIO/result/bart_large_cnn_dbs_dendrite/candidate_ranking')
    dbs_refs = os.path.expanduser('~/BRIO/result/bart_large_cnn_dbs_dendrite/reference_ranking')

    outputs = []
    for i in tqdm(range(n)):
        data_row = test[i]
        row = {'id': data_row['id'], 'article': data_row['article'], 'reference': data_row['highlights']}

        ea_cand_fn = os.path.join(ea_cands, f'{i}.dec')
        ea_ref_fn = os.path.join(ea_refs, f'{i}.ref')
        ea_pred = ea_preds.iloc[i]['from_extract_abstract'].split('<cand>')

        dbs_cand_fn = os.path.join(dbs_cands, f'{i}.dec')
        dbs_ref_fn = os.path.join(dbs_refs, f'{i}.ref')
        dbs_pred = dbs_preds.iloc[i]['abstract'].split('<cand>')

        ea_resolved, ea_tok, ea_conflicts = resolve(data_row, ea_cand_fn, ea_ref_fn, ea_pred)
        dbs_resolved, dbs_tok, dbs_conflicts = resolve(data_row, dbs_cand_fn, dbs_ref_fn, dbs_pred)

        row['ea'] = ea_resolved
        row['dbs'] = dbs_resolved

        row['ea_tok'] = ea_tok
        row['dbs_tok'] = dbs_tok

        row['ea_potential_conflicts'] = ea_conflicts
        row['dbs_potential_conflicts'] = dbs_conflicts

        outputs.append(row)
    outputs = pd.DataFrame(outputs)
    print(len(outputs[outputs['ea_potential_conflicts'] > 0]))
    print(len(outputs[outputs['dbs_potential_conflicts'] > 0]))
    outputs.to_csv(os.path.expanduser('~/cnndm_human_eval.csv'), index=False)
