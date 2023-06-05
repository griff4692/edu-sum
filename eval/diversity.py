import os

import argparse

import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
from tqdm import tqdm


def diversity_score(candidates):
    overlaps = []
    candidate_toks = list(map(lambda x: [y.lower() for y in nltk.word_tokenize(x) if len(y.strip()) > 0], candidates))
    diversities = []
    for i in range(len(candidates)):
        i_toks = set(list(map(lambda x: x.lower(), candidates[i].split(' '))))
        references = [c for j, c in enumerate(candidate_toks) if j != i]
        self_bleu = sentence_bleu(references, hypothesis=candidate_toks[i])
        diversities.append(1 - self_bleu)
        for j in range(i + 1, len(candidates)):
            j_toks = set(list(map(lambda x: x.lower(), candidates[j].split(' '))))
            avg_len = max(1., 0.5 * len(i_toks) + 0.5 * len(j_toks))
            overlap = len(i_toks.intersection(j_toks)) / avg_len
            overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)
    return diversities


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Score for Diversity')

    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('--experiment', default='add_doc')
    parser.add_argument('--fn', default='validation_sample_w_diverse_outputs')
    parser.add_argument('--candidate_column', default='extract')

    args = parser.parse_args()

    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    in_fn = os.path.join(results_dir, args.fn + '.csv')
    data_df = pd.read_csv(in_fn)
    candidates = data_df[args.candidate_column].fillna('NOPE').tolist()

    all_scores = []
    beam_scores = []
    for candidate_set in tqdm(candidates):
        if candidate_set == 'NOPE':
            all_scores.append(None)
            continue
        cand_list = candidate_set.split('<cand>')
        scores = diversity_score(cand_list)
        all_scores.append(np.mean(scores))
        print(f'Avg Diversity: {np.mean(all_scores)}')
        if len(beam_scores) == 0:
            beam_scores = [[] for _ in range(len(scores))]
        for beam, score, in enumerate(scores):
            beam_scores[beam].append(score)

    diversity_col = args.candidate_column + '_self_bleu'
    print(f'Adding column: {diversity_col}')
    data_df[diversity_col] = all_scores
    print(f'Re-saving output with new column {diversity_col} back to {in_fn}')
    data_df.to_csv(in_fn, index=False)

    for beam, score_arr in enumerate(scores):
        avg_beam_diversity = np.mean(score_arr)
        print(f'Beam {beam}: {avg_beam_diversity}')

    avg_diversity = np.mean(list(filter(None, all_scores)))
    print(f'Average Self-BLEU: {avg_diversity}')
