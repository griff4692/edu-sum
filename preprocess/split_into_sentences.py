import os

import argparse
import spacy
from datasets import load_dataset

from preprocess.extract_oracles import convert_to_sents
from sum_constants import summarization_name_mapping


def remove_non_ascii(text):
    return text.encode('ascii', errors='ignore').decode()


def sentencize(args, nlp, batch_data, input_col, target_col):
    batch_source_sents = [
        convert_to_sents(remove_non_ascii(inputs), nlp, is_dialogue=args.dataset == 'samsum')
        for inputs in batch_data[input_col]
    ]
    targets = list(map(remove_non_ascii, batch_data[target_col]))

    if args.dataset == 'cnn_dailymail':
        batch_target_sents = [
            [x.strip() for x in inputs.split('\n') if len(x.strip()) > 0] for inputs in targets
        ]
    else:
        batch_target_sents = [
            convert_to_sents(inputs, nlp, is_dialogue=args.dataset == 'samsum') for inputs in targets
        ]
        batch_target_sents = [
            [s.text.strip() for s in sentences] for sentences in batch_target_sents
        ]

    target_annotated = [
        ''.join(
            [f'<s{i}> {s.strip()}' for i, s in enumerate(target_sents)]
        ) for target_sents in batch_target_sents
    ]

    source_annotated = [
        ''.join(
            [f'<s{i}> {s.text.strip()}' for i, s in enumerate(source_sents) if i < args.max_num_sents]
        ) for source_sents in batch_source_sents
    ]

    return {
        'source_annotated': source_annotated,
        'target': targets,
        'target_annotated': target_annotated,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split dataset into sentences.')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('--num_proc', default=16, type=int)
    parser.add_argument('--max_num_sents', default=200, type=int)
    parser.add_argument('-debug', default=False, action='store_true')

    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    input_col, target_col = summarization_name_mapping[args.dataset]
    out_dir = os.path.join(args.data_dir, args.dataset + '_sentences')
    print(f'Saving to {out_dir}')

    print(f'Loading {args.dataset}...')
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset(args.dataset, '3.0.0')
    else:
        dataset = load_dataset(args.dataset)

    print('Loading Spacy...')
    nlp = spacy.load('en_core_web_sm')

    encoded_data = {}
    for split in args.splits.split(','):
        print(f'Processing {len(dataset[split])} {split} examples')
        encoded = dataset[split].map(lambda examples: sentencize(
            args, nlp, examples, input_col, target_col,
        ), batched=True, batch_size=100, num_proc=1 if args.debug else args.num_proc)
        encoded = encoded.filter(lambda example: len(example[input_col].strip()) > 0)
        encoded = encoded.add_column('dataset_idx', list(range(len(encoded))))
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
