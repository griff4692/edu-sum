import os
from glob import glob
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
from edu_gpt.eval_candidates import get_dataset_idx


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
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--num_candidates', default=16, type=int)

    # Configuration Parameters
    parser.add_argument('--extract_experiment', default=None)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--mode', default='vanilla')

    args = parser.parse_args()

    if args.experiment is None:
        args.experiment = args.mode + '_' + args.model + '_' + str(args.num_candidates)

    nlp = spacy.load('en_core_web_sm')
    data_fn = os.path.join(args.data_dir, args.dataset + '_edu_alignments')
    dataset = load_from_disk(data_fn)
    input_col, _ = summarization_name_mapping[args.dataset]
    split = 'test'

    results_dir = os.path.join(
        args.data_dir,
        'results',
        args.extract_experiment,
        args.experiment
    )

    print(results_dir)

    fns = list(glob(os.path.join(results_dir, '*.txt')))

    dataset_idxs = list(sorted(list(set([get_dataset_idx(x) for x in fns]))))

    test = dataset['test']
    idx2target = dict(zip(test['dataset_idx'], test['target']))

    agg_stats = []

    split_dataset = dataset[split]
    articles = split_dataset[input_col]
    brio_dataset = 'cnndm'
    brio_out_dir = os.path.expanduser(
        os.path.join('~', 'BRIO', brio_dataset, args.experiment)
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

        print(len(ptb_results), len(dataset_idxs))
        assert len(ptb_results) == len(dataset_idxs)

    for_ptb = []
    num_cand = -1
    for idx, dataset_idx in tqdm(enumerate(dataset_idxs), total=len(dataset_idxs)):
        dataset_idx = int(dataset_idx)
        idx_fns = list(sorted([fn for fn in fns if int(get_dataset_idx(fn)) == dataset_idx]))
        candidates = []
        for fn in idx_fns:
            with open(fn, 'r') as fd:
                candidates += [x.strip() for x in fd.read().split('<SEP>')]

        reference = idx2target[dataset_idx]

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
        ref_no_new_lower = ptb_prepare(reference)
        for_ptb += [ref_no_new_lower] + candidates_no_new_lower

        article_untok = articles[dataset_idx]
        reference_untok = reference

        article_untok, article_tok = brio_tokenize(article_untok, nlp)
        reference_untok, reference_tok = brio_tokenize(reference_untok, nlp)

        # Tokenize reference
        candidates_untok = []
        candidates_tok = []
        for cand_idx, cand_untok in enumerate(candidates):
            cand_untok, cand_tok = brio_tokenize(cand_untok, nlp)
            candidates_untok.append([cand_untok, 0.0])
            candidates_tok.append([cand_tok, 0.0])

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
            ptb_cands_w_rouges = [[c, 0.0] for c in ptb_cands]
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
