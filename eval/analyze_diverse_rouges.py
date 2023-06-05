import json

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


def analyze(experiment, output, summary_style=None, max_beams=16, use_calibration=False):
    score_col = 'calibrated_beam_score'
    if output.endswith('.csv'):
        suffix = ''
    else:
        suffix = '.csv'
    in_fn = f'/nlp/projects/faithsum/results/{experiment}/{output}{suffix}'
    print(in_fn)
    df = pd.read_csv(in_fn)
    reranked = score_col in df.columns and use_calibration
    print(f'Loaded {len(df)} examples')
    if summary_style is None:
        if 'from_extract_abstract' in df.columns:
            summary_style = 'from_extract_abstract'
        else:
            summary_style = 'abstract'

    df = df.dropna(subset=[summary_style])
    print(summary_style)
    df['source_len'] = df['source'].apply(lambda x: len(x.split(' ')))
    # orig_df = df.assign(bin=pd.qcut(df['source_len'], q=4, labels=np.arange(4)))

    if summary_style == 'abstract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'diversity'
    elif summary_style == 'implied_extract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'implied_diversity'
    elif summary_style == 'from_extract_abstract':
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = 'diversity'
    else:
        rouge_col = f'eval_{summary_style}_rouge1_f1'
        diversity_col = f'{summary_style}_diversity'

    rouges = [get_arr(x) for x in df[rouge_col].tolist()]
    cands = df[summary_style].tolist()

    try:
        diversities = [float(x) for x in df[diversity_col].dropna()]
    except:
        print('This has been fixed in generate but lets deal with it here.')
        diversities = []
        for x in df[diversity_col]:
            diversities.append(np.mean(json.loads(x)))
    n = len(df)

    rank_scores = None
    if reranked:
        rank_scores = [get_arr(x) for x in df[score_col].tolist()]

    avg_rouges = []
    max_rouges = []
    # avg_bartscores = []
    max_rouges_by_beam = [[] for _ in range(min(max_beams, len(rouges[0])))]
    avg_rouges_by_beam = [[] for _ in range(min(max_beams, len(rouges[0])))]
    avg_rouges_by_beam_cum = [[] for _ in range(min(max_beams, len(rouges[0])))]
    # avg_bartscores_by_beam = [[] for _ in range(len(rouges[0]))]

    lens_by_beam = [
        [] for _ in range(max_beams)
    ]

    avg_lens = []
    for i in tqdm(range(n)):
        rouge_arr = rouges[i]
        # bartscore_arr = bartscores[i]
        cand_arr = cands[i]
        rouge_arr_sorted = rouge_arr
        for beam, cand in enumerate(cand_arr.split('<cand>')):
            avg_lens.append(len(cand.split(' ')))
            lens_by_beam[beam].append(len(cand.split(' ')))

        # bartscore_arr_sorted = bartscore_arr
        if rank_scores is not None:
            scores = rank_scores[i]
            priority = np.argsort(-np.array(scores))
            rouge_arr_sorted = [rouge_arr[pidx] for pidx in priority]
            # bartscore_arr_sorted = [bartscore_arr[pidx] for pidx in priority]

        avg_rouges.append(np.mean(rouge_arr))
        # avg_bartscores.append(np.mean(bartscore_arr_sorted))
        max_rouges.append(max(rouge_arr))
        num_beams = min(max_beams, len(avg_rouges_by_beam))
        for beam in range(num_beams):
            try:
                cum_rouge = rouge_arr_sorted[:beam + 1]
                avg_rouges_by_beam[beam].append(rouge_arr_sorted[beam])
                avg_rouges_by_beam_cum[beam].append(np.mean(cum_rouge))
                max_rouges_by_beam[beam].append(max(cum_rouge))
            except Exception as e:
                print(str(e))
                print('If only happens once for nucleus that is fine because occasionally nucleus generates EM duplicates.')
            # cum_bartscore = bartscore_arr_sorted[:beam + 1]
            # avg_bartscores_by_beam[beam].append(np.mean(cum_bartscore))

    print(f'Mean Summary Length: {np.mean(avg_lens)}')
    print(f'Mean Avg inverse SELF-BLEU: {np.mean(diversities)}')
    print(f'Mean Avg ROUGE-1 F1: {np.mean(avg_rouges)}')
    print(f'Mean Max ROUGE-1 F1: {np.mean(max_rouges)}')
    # print(f'Mean Avg BartScore: {np.mean(avg_bartscores)}')
    print('Cumulative Mean Avg ROUGE-1 F1 by Beam...')
    out = []
    for beam in range(len(avg_rouges_by_beam)):
        out.append(str(np.mean(avg_rouges_by_beam_cum[beam])))
    print('\n'.join(out))

    print('Mean Avg ROUGE-1 F1 @ each Beam...')
    out = []
    for beam in range(len(avg_rouges_by_beam)):
        out.append(str(np.mean(avg_rouges_by_beam[beam])))
    print('\n'.join(out))

    print('Mean Max ROUGE-1 F1 by Beam...')
    out = []
    for beam in range(len(max_rouges_by_beam)):
        out.append(str(np.mean(max_rouges_by_beam[beam])))
    print('\n'.join(out))

    print('Summary Length by Beam')
    for beam in range(len(lens_by_beam)):
        print(beam, np.mean(lens_by_beam[beam]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze ROUGE candidates.')
    parser.add_argument('--experiment', default='cnn_e_final')
    parser.add_argument('--fn', default='test_from_beam_16_extract_cnn_ea_final_mle_10.0_like_1_unlike_1')

    args = parser.parse_args()

    analyze(args.experiment, args.fn)
