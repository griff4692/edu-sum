from collections import defaultdict

import pandas as pd
import argparse
from p_tqdm import p_uimap

from eval.rouge_metric import RougeMetric
from eval.analyze_diverse_rouges import analyze


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return [float(y) for y in num_str.split(delim)]


def evaluate_summary(rouge_metric, generated, gold, prefix=''):
    outputs = rouge_metric.evaluate_batch([generated], [gold], aggregate=True)['rouge']
    stats = {}
    for rouge_type in ['1', '2', 'L']:
        fscore = outputs[f'rouge_{rouge_type.lower()}_f_score']
        stats[f'{prefix}rouge{rouge_type}_precision'] = outputs[f'rouge_{rouge_type.lower()}_precision']
        stats[f'{prefix}rouge{rouge_type}_recall'] = outputs[f'rouge_{rouge_type.lower()}_recall']
        stats[f'{prefix}rouge{rouge_type}_f1'] = fscore
    return stats


def process_example(record, rouge_metric):
    try:
        metric_str = defaultdict(list)
        for col in record:
            if col not in {'abstract', 'implied_extract', 'extract', 'from_extract_abstract'}:
                continue
            if type(record[col]) == float:
                continue
            summaries = record[col].split('<cand>')
            metrics = list(map(lambda s: evaluate_summary(rouge_metric, s, record['reference']), summaries))
            mean_f1s = []
            for m in metrics:
                for k, v in m.items():
                    metric_str['eval' + '_' + col + '_' + k].append(str(v))
                mean_f1 = (m['rouge1_f1'] + m['rouge2_f1'] + m['rougeL_f1']) / 3.0
                mean_f1s.append(mean_f1)
                metric_str['eval' + '_' + col + '_mean_f1'].append(str(mean_f1))

            # existing_rouges = get_arr(record[col + '_rouges'])
            # corel = spearmanr(existing_rouges, mean_f1s)[0]
        for k in metric_str:
            metric_str[k] = ','.join(metric_str[k])
        record.update(metric_str)
    except Exception as e:
        print('Issue parsing example: ', str(e))
    return record


def process_file(fn, num_cpus=32, should_analyze=True):
    rouge_metric = RougeMetric()

    print(f'Reading predictions from {fn}')
    outputs = pd.read_csv(fn)
    records = outputs.to_dict('records')

    if num_cpus == 1:
        augmented_records = list(map(lambda record: process_example(record, rouge_metric), records))
    else:
        augmented_records = p_uimap(
            lambda record: process_example(record, rouge_metric), records, num_cpus=num_cpus
        )
    augmented_df = pd.DataFrame(augmented_records).sort_values(by='dataset_idx').reset_index(drop=True)
    print(f'Saving with PERL eval ROUGE columns added back to {fn}')
    augmented_df.to_csv(fn, index=False)

    for col in augmented_df.columns.tolist():
        if 'rouge' in col and 'eval' in col:
            try:
                print(col, ': ', str(augmented_df[col].dropna().mean()))
            except:
                print(f'Array col: {col}')

    s = fn.split('/')
    if should_analyze:
        analyze(s[-2], s[-1].replace('.csv', ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADD Python PERL ROUGE Eval')

    parser.add_argument('--fn')
    parser.add_argument('--num_cpus', default=32, type=int)

    args = parser.parse_args()

    process_file(args.fn, args.num_cpus)
