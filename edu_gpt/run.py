import os
from glob import glob

import regex as re
import argparse
import numpy as np
import openai
from tqdm import tqdm
import backoff

from oa_secrets import OA_KEY, OA_ORGANIZATION


openai.organization = OA_ORGANIZATION
openai.api_key = OA_KEY


def shorten(prompt):
    batches = prompt.split('-----')
    return '-----'.join([batches[0], batches[-1]])


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=25)
def chat_gpt(args, messages, model='gpt-4', top_p=None, temperature=0.3, max_tokens=256):
    if top_p is None:
        response = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            n=args.candidates
        )
    else:
        response = openai.ChatCompletion.create(
            model=model, messages=messages, top_p=top_p, max_tokens=max_tokens,
            n=args.candidates
        )
    return [choice.message.content for choice in response['choices']]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def instruct_gpt(prompt, model='gpt-4', top_p=None, temperature=0.3, max_tokens=256):
    if top_p is None:
        response = openai.Completion.create(
            model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens,
            n=args.candidates
        )
    else:
        response = openai.Completion.create(
            model=model, prompt=prompt, top_p=top_p, max_tokens=max_tokens,
            n=args.candidates
        )
    return [choice.text for choice in response['choices']]


def get_dataset_idx(x):
    return x.split('/')[-1].replace('.txt', '').split('_')[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-3.5 / GPT-4.5')

    # Configuration Parameters
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--extract_experiment', default='cnn_e_final')
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['text-davinci-003', 'gpt-3.5-turbo', 'gpt-4'])
    parser.add_argument('--max_examples', default=1000, type=int)
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--mode', default='pga', choices=['vanilla', 'pga'])
    parser.add_argument('--candidates', default=16, type=int)
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--top_p', default=None, type=float)

    args = parser.parse_args()

    in_dir = os.path.join(os.environ['DATA_DIR'], 'results', args.extract_experiment, args.mode + '_prompts')
    assert os.path.exists(in_dir)
    if args.experiment is None:
        args.experiment = args.mode + '_' + args.model + '_' + str(args.candidates)
    out_dir = os.path.join('/nlp/projects/faithsum/results', args.extract_experiment, args.experiment)
    print(f'Saving to {out_dir}.')
    os.makedirs(out_dir, exist_ok=True)

    is_chat = args.model in {'gpt-3.5-turbo', 'gpt-4'}

    fns = list(glob(os.path.join(in_dir, '*.txt')))

    ids = list(sorted(list(set([get_dataset_idx(x) for x in fns]))))
    if args.max_examples is not None and len(ids) > args.max_examples:
        np.random.seed(1992)
        np.random.shuffle(ids)
        ids = ids[:args.max_examples]
    ids = set(ids)
    fns_filt = [fn for fn in fns if get_dataset_idx(fn) in ids]

    for fn in tqdm(fns_filt):
        out_fn = os.path.join(out_dir, fn.split('/')[-1])
        if os.path.exists(out_fn) and not args.overwrite:
            print(f'Warning! Skipping: {out_fn}. Run with -overwrite if not desired behavior.')
            continue
        with open(fn, 'r') as fd:
            prompt = fd.read()
            # Ultimately can replace this
            prompt = re.sub(r'\s*<e>\s*', ' <e> ', prompt).strip()
            prompt = re.sub(r'\s*</e>\s*', ' </e> ', prompt).strip()

            num_toks = len(prompt.split(' '))
            if num_toks > 2750:  # This will likely trigger 4k token limit error
                prompt = shorten(prompt)
                new_toks = len(prompt.split(' '))
                print(f'Shortened from {num_toks} to {new_toks}')

            if is_chat:
                messages = [
                    # Boost its ego first
                    {'role': 'system', 'content': 'You are a helpful and concise assistant for text summarization.'},
                    {'role': 'user', 'content': prompt}
                ]
                output = chat_gpt(
                    args, messages=messages, top_p=args.top_p, temperature=args.temperature, model=args.model
                )
            else:
                output = instruct_gpt(
                    args, prompt=prompt, top_p=args.top_p, temperature=args.temperature, model=args.model
                )

            with open(out_fn, 'w') as fd:
                fd.write('<SEP>'.join(output))
