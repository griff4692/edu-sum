import os
import pandas as pd
import ujson
from tqdm import tqdm
import numpy as np
import regex as re

import argparse
from datasets import load_from_disk
import spacy
from nltk import sent_tokenize

from sum_constants import summarization_name_mapping
from preprocess.extract_oracles import dialogue_to_sents


def brio_tokenize(text, nlp):
    doc = nlp(text)
    sents_untok = list(doc.sents)
    sents_untok = [str(x).strip() for x in sents_untok if len(str(x).strip()) > 0]
    sents_tok = [' '.join([str(y) for y in x]) for x in doc.sents]
    sents_tok = [x.strip() for x in sents_tok if len(x.strip()) > 0]
    return sents_untok, sents_tok


def brio_samsum_tokenize(text, nlp):
    sents_untok = dialogue_to_sents(text, nlp)
    sents_untok_str = [str(x).strip() for x in sents_untok if len(str(x).strip()) > 0]
    sents_tok = [' '.join([str(y) for y in x]) for x in sents_untok]
    sents_tok = [x.strip() for x in sents_tok if len(x.strip()) > 0]
    return sents_untok_str, sents_tok


def get_arr(num_str):
    if '<cand>' in num_str:
        delim = '<cand>'
    else:
        delim = ','
    return np.array([float(y) for y in num_str.split(delim)])


def ptb_prepare(text):
    return re.sub('\s+', ' ', re.sub(r'\n', ' ', text)).lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert fn to BRIO directory format')
    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('--experiment', default='cnn_e_final')
    parser.add_argument('--save_suffix', default=None, required=True)
    parser.add_argument('--fn', default='test_from_beam_16_extract.csv')
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--splits', default='test')
    parser.add_argument('--num_candidates', default=16, type=int)

    args = parser.parse_args()

    if not args.fn.endswith('.csv'):
        args.fn += '.csv'

    nlp = spacy.load('en_core_web_sm')
    data_fn = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    dataset = load_from_disk(data_fn)
    input_col, _ = summarization_name_mapping[args.dataset]
    for split in args.splits.split(','):
        if '{}' in args.fn:
            fn = os.path.join(args.data_dir, 'results', args.experiment, args.fn.format(split))
        else:
            fn = os.path.join(args.data_dir, 'results', args.experiment, args.fn)
        print(f'Reading data from {fn}')
        df = pd.read_csv(fn)
        split_dataset = dataset[split]
        articles = split_dataset[input_col]
        records = df.to_dict('records')
        brio_dataset = None
        if args.dataset == 'cnn_dailymail':
            brio_dataset = 'cnndm'
        else:
            brio_dataset = args.dataset
        brio_out_dir = os.path.expanduser(
            os.path.join('~', 'BRIO', brio_dataset, args.experiment + '_' + args.save_suffix)
        )
        split_dir = os.path.join(brio_out_dir, 'diverse', split)
        print(f'Will save outputs to {split_dir}')
        os.makedirs(split_dir, exist_ok=True)

        ptb_results = None
        ptb_fn = os.path.join(brio_out_dir, split + '.tokenized')
        if os.path.exists(ptb_fn):
            print(f'Loading PTB tokenized results from {ptb_fn}')
            ptb_results = []
            with open(ptb_fn, 'r') as fd:
                lines = fd.readlines()
                chunksize = args.num_candidates + 1
                for start in range(0, len(lines), chunksize):
                    ref_tok = lines[start].strip()
                    cand_tok = [x.strip() for x in lines[start + 1:start + 1 + args.num_candidates]]
                    assert len(cand_tok) == args.num_candidates
                    ptb_results.append({'ref': ref_tok, 'cand': cand_tok})

            print(len(ptb_results), len(records))
            assert len(ptb_results) == len(records)

        for_ptb = []
        num_cand = -1
        for idx, record in tqdm(enumerate(records), total=len(records)):
            dataset_idx = record['dataset_idx']
            prediction_col = 'from_extract_abstract' if 'from_extract_abstract' in record else 'abstract'
            candidates = record[prediction_col].split('<cand>')

            if num_cand == -1:
                num_cand = len(candidates)
            try:
                num_repeated = 0
                assert num_cand == len(candidates) == args.num_candidates
            except:
                print(num_cand, len(candidates), args.num_candidates)
                print(f'Dataset Idx={dataset_idx} does not have enough unique candidates. Duplicating for now.')
                last_cand = candidates[-1]
                num_repeated = num_cand - len(candidates)
                for _ in range(num_repeated):
                    candidates.append(last_cand)

            candidates_no_new_lower = [ptb_prepare(c) for c in candidates]
            ref_no_new_lower = ptb_prepare(record['reference'])
            for_ptb += [ref_no_new_lower] + candidates_no_new_lower

            article_untok = articles[dataset_idx]
            reference_untok = record['reference']
            if prediction_col == 'from_extract_abstract':
                if 'eval_from_extract_abstract_rouge1_f1' in record:
                    rouges = get_arr(record['eval_from_extract_abstract_rouge1_f1'])
                else:
                    rouges = get_arr(record['from_extract_rouges'])
            else:
                if 'eval_abstract_rouge1_f1' in record:
                    rouges = get_arr(record['eval_abstract_rouge1_f1'])
                else:
                    if args.num_candidates == 1:
                        rouges = [record['rouge1_f1']]
                    else:
                        rouges = get_arr(record['abstract_rouges'])
            if num_repeated > 0:
                last = rouges[-1]
                extra = np.array([last for _ in range(num_repeated)])
                rouges = np.concatenate((rouges, extra))

            if args.dataset == 'samsum':
                article_untok, article_tok = brio_samsum_tokenize(article_untok, nlp)
                reference_untok, reference_tok = brio_samsum_tokenize(reference_untok, nlp)
            else:
                article_untok, article_tok = brio_tokenize(article_untok, nlp)
                reference_untok, reference_tok = brio_tokenize(reference_untok, nlp)

            # Tokenize reference
            candidates_untok = []
            candidates_tok = []
            for cand_idx, cand_untok in enumerate(candidates):
                cand_untok, cand_tok = brio_tokenize(cand_untok, nlp)
                rouge = rouges[cand_idx]
                candidates_untok.append([cand_untok, rouge])
                candidates_tok.append([cand_tok, rouge])

            obj = {
                'article': article_tok,
                'article_untok': article_untok,
                'abstract_untok': reference_untok,
                'candidates_untok': candidates_untok,
            }

            if ptb_results is None:
                toks = {
                    'abstract': reference_tok,
                    'candidates': candidates_tok,
                }
            else:
                try:
                    ptb_ref = sent_tokenize(ptb_results[idx]['ref'])
                except:
                    ptb_ref = [ptb_results[idx]['ref']]

                ptb_cands = []
                for cand in ptb_results[idx]['cand']:
                    try:
                        tok = sent_tokenize(cand)
                    except:
                        print(f'Failed to tokenize sentence: {cand}')
                        tok = [cand]
                    ptb_cands.append(tok)
                ptb_cands_w_rouges = [[c, r] for c, r in zip(ptb_cands, rouges)]
                toks = {
                    'abstract': ptb_ref,
                    'candidates': ptb_cands_w_rouges,
                }

            obj.update(toks)

            out_fn = os.path.join(split_dir, f'{idx}.json')
            with open(out_fn, 'w') as fd:
                ujson.dump(obj, fd)

        print(f'Saved BRIO outputs to {split_dir}')
        to_ptb_fn = os.path.join(brio_out_dir, split + '.txt')
        if not os.path.exists(to_ptb_fn):
            with open(to_ptb_fn, 'w') as fd:
                fd.write('\n'.join(for_ptb))
