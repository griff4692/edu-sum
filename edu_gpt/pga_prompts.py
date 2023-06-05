import os

from datasets import load_from_disk
import argparse
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from p_tqdm import p_uimap
from model.gen_from_extract import filter_out_extract_tags


INSTRUCTION = 'Summarize the content in between the HTML tags <e> and </e> in one to three sentences.'


def dump(record, icl, source):
    id = record['dataset_idx']

    for ct, cand_idx_str in enumerate(record['extract_idx'].split('<cand>')):
        few_shot_str = sample_few_shot(icl, k=args.k)
        cand_idx = [int(x) for x in cand_idx_str.split(',')]
        gpt_input = filter_out_extract_tags(source, cand_idx)
        prompt = f'{INSTRUCTION}\n\n{few_shot_str}\n\n-----\n\nArticle: {gpt_input}\n\nFocused Summary: '
        out_fn = os.path.join(out_dir, f'{id}_{ct}.txt')

        with open(out_fn, 'w') as fd:
            fd.write(prompt)


def sample_few_shot(icl, k=3):
    idxs = np.random.choice(np.arange(len(icl)), size=(k, ), replace=False)
    icl_sample = icl.select(idxs)
    few_shot = []

    for example in icl_sample:
        source = example['source_edu_annotated']
        target = example['target']
        gpt_input = filter_out_extract_tags(source, example['oracle_idxs'])
        prompt = f'Article: {gpt_input}\n\nFocused Summary: {target}'
        few_shot.append(prompt)

    few_shot_str = '\n\n-----\n\n'.join(few_shot)
    return few_shot_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Plan-Guided Summaries for GPT-3.5 / 4')
    parser.add_argument('--plan_rouge1_threshold', default=0.75)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument(
        '--extract_fn', default='/nlp/projects/faithsum/results/cnn_e_final/test_beam_16_outputs_1k_sample.csv'
    )

    args = parser.parse_args()

    out_dir = '/'.join(args.extract_fn.split('/')[:-1]) + '/pga_prompts'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.extract_fn)
    records = df.to_dict('records')

    dataset = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')
    train = dataset['train']
    icl = train.filter(
        lambda example:
        example['oracle_best_rouge1'] >= args.plan_rouge1_threshold and len(sent_tokenize(example['highlights'])) < 5
        and len(example['source_edu_annotated'].split(' ')) < 512,
        num_proc=16
    )
    print(f'Filtered out. {len(icl) / len(train)} remaining.')

    test = dataset['test']
    idx2source = dict(zip(test['dataset_idx'], test['source_edu_annotated']))

    list(p_uimap(lambda record: dump(record, icl, idx2source[record['dataset_idx']]), records))

    print(f'Finished. Saved to {out_dir}')
