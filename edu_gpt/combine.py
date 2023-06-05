from collections import defaultdict
import os
import argparse
from glob import glob
import pandas as pd

from evaluate import load
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import regex as re


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


def add_rouge(obj, reference, rouge):
    pred = obj['prediction']
    robj = rouge.compute(predictions=[pred], references=[reference], use_aggregator=False)
    obj['rouge1'] = robj['rouge1'][0]
    obj['rouge2'] = robj['rouge2'][0]
    obj['rougeL'] = robj['rougeL'][0]
    obj['rougeLsum'] = robj['rougeLsum'][0]
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine LLM diverse candidates')

    # Configuration Parameters
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--extract_experiment', default='cnn_e_final')
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo', 'gpt-4'])

    args = parser.parse_args()

    dataset = []

    rouge = load('rouge', keep_in_memory=True)

    experiments = [
        ('pga_gpt-3.5-turbo', 'pga'),
        ('vanilla_gpt-3.5-turbo', 'single'),
        ('vanilla_gpt-3.5-turbo_16', 'temperature'),
        ('vanilla_gpt-3.5-turbo_16_nucleus', 'nucleus'),
    ]

    pga_prompts = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, 'pga_prompts')
    vanilla_prompts = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, 'vanilla_prompts')

    id2cands = defaultdict(list)
    id2meta = defaultdict(dict)

    experiment2fns = {}

    for experiment, _ in experiments:
        response_dir = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, experiment)
        experiment2fns[experiment] = list(glob(os.path.join(response_dir, '*.txt')))

    df = pd.read_csv('/nlp/projects/faithsum/results/cnn_e_final/test_beam_16_outputs_1k_sample.csv')

    test = load_from_disk('/nlp/projects/faithsum/cnn_dailymail_edu_alignments')['test']

    out = []
    for record in tqdm(df.to_dict('records')):
        dataset_idx = record['dataset_idx']
        data_record = test[dataset_idx]
        van_fn = os.path.join(vanilla_prompts, f'{dataset_idx}.txt')
        with open(van_fn, 'r') as fd:
            vanilla_prompt = fd.read().strip()

        row = {
            'id': data_record['id'],
            'source': record['source'],
            'source_edu_annotated': data_record['source_edu_annotated'],
            'reference': record['reference'],
            'candidates': [],
            'vanilla_prompt': vanilla_prompt,
            'pga_prompts': [],
            'pga_edu_extract_idxs': [[int(y) for y in get_arr(x)] for x in record['extract_idx'].split('<cand>')],
        }
        cands = []
        for experiment, name in experiments:
            if name == 'pga':
                for cand_idx in range(16):
                    in_fn = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, experiment, f'{dataset_idx}_{cand_idx}.txt')

                    pga_fn = os.path.join(pga_prompts, f'{dataset_idx}_{cand_idx}.txt')
                    with open(pga_fn, 'r') as fd:
                        pga_prompt = fd.read().strip()
                        # Ultimately can replace this
                        pga_prompt = re.sub(r'\s*<e>\s*', ' <e> ', pga_prompt).strip()
                        pga_prompt = re.sub(r'\s*</e>\s*', ' </e> ', pga_prompt).strip()
                        row['pga_prompts'].append(pga_prompt)

                    with open(in_fn, 'r') as fd:
                        cand = fd.read().strip()
                        row['candidates'].append(add_rouge({
                            'prediction': cand,
                            'method': 'pga',
                            'method_beam': cand_idx + 1,
                        }, row['reference'], rouge))
            else:
                in_fn = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, experiment, f'{dataset_idx}.txt')
                with open(in_fn, 'r') as fd:
                    cand = fd.read().strip().split('<SEP>')

                for i, c in enumerate(cand):
                    row['candidates'].append(add_rouge({
                        'prediction': c,
                        'method': name,
                        'method_beam': i + 1,
                    }, row['reference'], rouge))

        out.append(row)

    dataset = Dataset.from_list(out)
    dataset.save_to_disk('/nlp/projects/faithsum/cnn_dailymail_diverse_gpt3')
