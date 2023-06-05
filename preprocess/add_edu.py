import os
import regex as re

import argparse
import ujson
from datasets import load_from_disk
from transformers import AutoTokenizer
from model.model_utils import infer_hf_model


EDU_SPECIAL_TOKENS = ['<e>', '</e>']


def list_rindex(li, x):
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError(f'{x} is not in list')


def ensure_has_edu(text):
    ct = len(re.findall('<e>', text))
    if ct == 0:
        return f'<e>{text}</e>'
    return text


def add_edus_and_ids(args, split, tokenizer, batch_data, max_input_length=1024, max_output_length=256):
    SOE, EOE = tokenizer.additional_special_tokens_ids

    target_edu_annotated = []
    source_edu_annotated = []

    for id in batch_data['id']:
        fn = os.path.join(args.data_dir, 'edu', args.dataset, split, f'{id}.json')
        assert os.path.exists(fn)
        with open(fn, 'r') as fd:
            edus = ujson.load(fd)

        sedu = [x for x in edus if x['type'] == 'source']
        tedu = [x for x in edus if x['type'] == 'target']

        source_sents_w_edu = list(sorted(sedu, key=lambda x: x['sent_idx']))
        target_sents_w_edu = list(sorted(tedu, key=lambda x: x['sent_idx']))

        flat_source_sents_w_edu = ' '.join(list(map(lambda x: ensure_has_edu(x['sent_w_edu']), source_sents_w_edu)))
        target_edu_annotated.append(' '.join(list(map(lambda x: ensure_has_edu(x['sent_w_edu']), target_sents_w_edu))))
        source_edu_annotated.append(flat_source_sents_w_edu)

    input_ids = tokenizer(
        source_edu_annotated,
        truncation=True,
        max_length=max_input_length,
    )['input_ids']

    input_ids_fixed = []
    # Two things to correct for
    for id_seq in input_ids:
        if id_seq[-1] == SOE:  # If the sequence ends in a start EDU token lets remove that start token (shift back)
            input_ids_fixed.append(id_seq[:-1])  # Remove that
        elif list_rindex(id_seq, SOE) > list_rindex(id_seq, EOE):
            assert len([x for x in id_seq if x == SOE]) == len([x for x in id_seq if x == EOE]) + 1
            # Let's convert last token to EOE
            id_seq[-1] = EOE
            input_ids_fixed.append(id_seq)
        else:
            input_ids_fixed.append(id_seq)

    decoded = tokenizer.batch_decode(input_ids_fixed, skip_special_tokens=False)

    num_source_edus_post_trunc_start = [
        len(re.findall('<e>', x)) for x in decoded
    ]

    num_source_edus_post_trunc = [
        len(re.findall('</e>', x)) for x in decoded
    ]

    assert all([
        a == b for a, b in zip(num_source_edus_post_trunc_start, num_source_edus_post_trunc)
    ])

    labels = tokenizer(
        batch_data['target'],
        truncation=True,
        max_length=max_output_length,
    )['input_ids']

    row = {
        'source_edu_annotated': source_edu_annotated,
        'target_edu_annotated': target_edu_annotated,
        'input_ids': input_ids_fixed,
        'labels': labels,
        # Tokenizer truncates > 1,024 token sources. We just record the pre and post trunc \# of EDUs
        # We will only compute oracle alignments up to truncated
        'num_edus_post_trunc': num_source_edus_post_trunc,
    }

    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add EDUs to the Sentencized Datasets')

    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='train,validation,test')
    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--num_proc', default=64, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-use_pegasus', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.num_proc = 1

    if args.use_pegasus:
        pegasus_suffix = '_pegasus'
        args.hf_model = 'google/pegasus-xsum'
    else:
        pegasus_suffix = ''
        infer_hf_model(args, is_abstract=False)

    out_dir = os.path.join(args.data_dir, args.dataset + f'_edus{pegasus_suffix}')
    print(f'Saving to {out_dir}')

    if args.dataset == 'xsum':
        max_input_length = 512
        max_output_length = 64
    else:
        max_input_length = 1024
        max_output_length = 256

    print(f'Loading {args.dataset}...')
    sent_dir = os.path.join(args.data_dir, args.dataset + '_sentences')
    dataset = load_from_disk(sent_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
    # Start and End of EDU spans
    special_tokens_dict = {'additional_special_tokens': EDU_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    encoded_data = {}
    for split in args.splits.split(','):
        # Filter dataset by which have been extracted
        filtered = dataset[split].filter(
            lambda ex: os.path.exists(os.path.join(args.data_dir, 'edu', args.dataset, split, ex['id'] + '.json')),
            batched=False, num_proc=args.num_proc,
        )

        print(f'Processing {len(filtered)}/{len(dataset[split])} {split} examples')
        encoded = filtered.map(
            lambda examples: add_edus_and_ids(
                args, split, tokenizer, examples, max_input_length=max_input_length,
                max_output_length=max_output_length,
            ),
            batched=True, batch_size=1000, num_proc=args.num_proc,
            remove_columns=['source_annotated', 'target_annotated'],
        )
        dataset[split] = encoded
    dataset.save_to_disk(out_dir)
