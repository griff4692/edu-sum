import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BartTokenizer

from data_utils import get_path_from_exp, infer_dataset
from model.main import add_edus_to_tokenizer
from model.dataset import SummaryDataModule
from model.model import TransformerSummarizer
from model.model_utils import infer_hf_model
from global_utils import get_free_gpus


DATASET_KWARGS = {
    'cnn_dailymail': {
        'abstract': {'min_length': 56, 'max_length': 142, 'length_penalty': 2.0},
        'extract': {'min_length': 3, 'max_length': 20, 'length_penalty': 1.0},
    },
    'xsum': {
        'extract': {'min_length': 2, 'max_length': 20, 'length_penalty': 2.0},
        'abstract': {'min_length': 11, 'max_length': 62, 'length_penalty': 0.6},
    },
    'nyt': {
        'abstract': {'min_length': 56, 'max_length': 256, 'length_penalty': 2.0},
        'extract': {'min_length': 3, 'max_length': 25, 'length_penalty': 1.0},
    }
}


BEAM_KWARGS = {
    # https://discuss.huggingface.co/t/facebook-bart-large-cnn-has-a-low-rouge-score-on-cnn-dailymail/673/2
    'num_beams': 4,  # Over-ridden by num_return_sequences
}

DIVERSE_KWARGS = {
    'cnn_dailymail': {'diversity_penalty': 1.0,},
    'nyt': {'diversity_penalty': 1.0},
    'xsum': {'diversity_penalty': 1.0},  # 0.1 performed worse
}

NUCLEUS_KWARGS = {
    'num_beams': 4,
    'top_p': 0.92,
    'top_k': 0,
    'do_sample': True,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BART/PEGASUS Generator & Evaluator.')
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR', '~/tmp'))
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-do_not_save', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--device', default=None, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--per_device_eval_bs', type=int, default=1)
    # Beam Search or Nucleus Sampling (more diverse)
    parser.add_argument('-add_sent_toks', default=False, action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=1956, type=int)
    parser.add_argument('--max_num_sents', type=int, default=200)
    parser.add_argument('--extract_method', type=str, default='generate', choices=['generate', 'select'])
    parser.add_argument('-use_hf_rouge', default=False, action='store_true')  # Much faster to use HF implementation
    parser.add_argument('--bootstraps', default=1, type=int)
    parser.add_argument('-add_brio_loss', default=False, action='store_true')
    parser.add_argument('-extract_indicators', default=False, action='store_true')
    parser.add_argument(
        '--summary_style',
        default='extract',
        choices=[
            'extract_abstract',
            'abstract',
            'extract'
        ], help='Target output during training. plan is a sequence of <s{idx}> tokens, extract is oracle summary, '
                'abstract is original reference'
    )
    parser.add_argument('--hf_model', default=None)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--num_chunks', default=8, type=int)

    # Decoding Parameters
    parser.add_argument('--decode_method', default='beam', choices=['beam', 'diverse', 'nucleus'])
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--length_penalty', default=None, type=float)
    parser.add_argument('--diversity_penalty', default=None, type=float)
    parser.add_argument('--copy_bart_class_dropout', default=0.1, type=float)

    args = parser.parse_args()

    args.add_sent_toks = args.add_sent_toks or 'extract' in args.summary_style or args.extract_indicators

    np.random.seed(args.seed)

    # These are ignored but need to pass something in
    args.per_device_eval_bs = -1
    args.per_device_train_bs = -1

    if args.wandb_name is None:
        assert args.experiment is not None

    if args.experiment is None:
        args.experiment = args.wandb_name

    infer_dataset(args, 'wandb_name')
    infer_hf_model(args, is_abstract=args.summary_style == 'abstract')

    if args.dataset == 'xsum':
        args.max_input_length = 512
    else:
        args.max_input_length = 1024

    weight_dir = os.path.join(args.data_dir, 'weights')
    results_dir = os.path.join(args.data_dir, 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    if args.cpu:
        gpu = 'cpu'
    else:
        gpu = free_gpus[0] if args.device is None else args.device

    # Generating from this pre-trained model
    if args.wandb_name is None:
        print('Warning! Loading in pre-trained weights. Specify --wandb_name if you want to load fine-tuned weights.')
        assert args.summary_style == 'abstract'
        args.lr = 1.0   # Needed to load
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)
        add_edus_to_tokenizer(tokenizer)
        model = TransformerSummarizer(args, tokenizer=tokenizer, hf_model=args.hf_model).to(gpu).eval()
    else:
        ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)
        tokenizer_dir = os.path.join(weight_dir, args.wandb_name, 'tokenizer')
        print(f'Loading tokenizer from {tokenizer_dir}...')
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)
        except:  # BRIO model doesn't load from AutoTokenizer
            tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_dir)

        print(f'Loading model from {ckpt_path}...')
        args.lr = None
        model = TransformerSummarizer.load_from_checkpoint(
            args=args, checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False
        ).to(gpu).eval()

    model.hparams.summary_style = args.summary_style
    model.hparams.extract_indicators = args.extract_indicators
    datamodule = SummaryDataModule(args, tokenizer)
    model.on_predict_start()

    exp_results = []
    for exp_id in range(args.bootstraps):
        # Override behavior during training
        dataloader_kwargs = {'shuffle': False, 'batch_size': args.batch_size}

        if args.split == 'train' and args.chunk is not None:
            dataloader, dataset_idxs = datamodule.get_train_chunk(
                args.chunk, args.num_chunks, **dataloader_kwargs
            )
        else:
            dataloader, dataset_idxs = datamodule.get_split(
                args.split, max_examples=args.max_examples, **dataloader_kwargs
            )
        outputs = []

        gen_kwargs = DATASET_KWARGS[args.dataset][args.summary_style]
        if args.length_penalty is not None:
            default_lp = gen_kwargs['length_penalty']
            print(f'Changing length penalty from default of {default_lp} to {args.length_penalty}')
            gen_kwargs['length_penalty'] = args.length_penalty
        gen_kwargs['use_hf_rouge'] = args.use_hf_rouge
        gen_kwargs['num_return_sequences'] = args.num_return_sequences

        if args.num_return_sequences == 1:
            assert args.decode_method == 'beam'
            if args.dataset == 'xsum':
                BEAM_KWARGS['num_beams'] = 8
            gen_kwargs.update(BEAM_KWARGS)
        else:
            if args.decode_method == 'beam':
                gen_kwargs.update(BEAM_KWARGS)
                if args.num_return_sequences > gen_kwargs['num_beams']:
                    gen_kwargs['num_beams'] = args.num_return_sequences
            elif args.decode_method == 'diverse':
                gen_kwargs.update(DIVERSE_KWARGS[args.dataset])
                if args.diversity_penalty is not None:
                    default_dp = gen_kwargs['diversity_penalty']
                    print(f'Changing diversity penalty from default of {default_dp} to {args.diversity_penalty}')
                    gen_kwargs['diversity_penalty'] = args.diversity_penalty
                gen_kwargs['num_beam_groups'] = args.num_return_sequences
                gen_kwargs['num_beams'] = args.num_return_sequences
            else:
                gen_kwargs.update(NUCLEUS_KWARGS)

        print(gen_kwargs)
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
            start = args.batch_size * batch_idx
            actual_batch_size = len(batch['input_ids'])
            end = start + actual_batch_size
            batch_dataset_idxs = dataset_idxs[start:end]
            with torch.no_grad():
                batch_stats = model.predict_step(batch, **gen_kwargs)
                for j in range(actual_batch_size):
                    batch_stats[j]['dataset_idx'] = batch_dataset_idxs[j]
                outputs += batch_stats

        outputs = pd.DataFrame(outputs)
        decode_suffix = args.decode_method + '_' + str(args.num_return_sequences)
        chunk_suffix = '' if args.chunk is None else f'_chunk_{args.chunk}'
        out_fn = os.path.join(results_dir, f'{args.split}_{decode_suffix}_outputs{chunk_suffix}.csv')

        if not args.do_not_save:
            print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
            outputs.to_csv(out_fn, index=False)
        num_col = outputs.select_dtypes('number')
        for col in list(num_col.columns):
            print(f'{col}: {num_col[col].dropna().mean()}')

        table_cols = [
            'rouge1_f1',
            'rouge2_f1',
            'rougeL_f1',
            'extract_rouge1_f1',
            'extract_rouge2_f1',
            'extract_rougeL_f1',
        ]

        out_str = ''
        for col in table_cols:
            try:
                v = outputs[col].dropna().mean()
                out_str += f'{round(v, 3)},'
            except:  # If the column doesn't exist (i.e., we are generating for an abstractive model, extract_ won't exist)
                out_str += ','
        out_str.strip(',')
        # print(','.join(table_cols))
        # print(out_str)

        agg_cols = [
            'rouge1_f1', 'extract_rouge1_f1',
            'best_extract_rouge1_f1', 'best_abstract_rouge1_f1',
            'avg_rouge1_f1', 'avg_extract_rouge1_f1',
            'diversity', 'extract_diversity'
        ]

        exp_row = {
            col: outputs[col].dropna().mean() for col in agg_cols if col in list(outputs.columns)
        }

        exp_results.append(exp_row)

        if args.summary_style == 'extract':
            extract_idxs = outputs['extract_idx'].tolist()
            lens = []
            for cands in extract_idxs:
                for cand in cands.split('<cand>'):
                    lens.append(len(cand.split(',')))
            print(f'Average extract length: {np.mean(lens)}')

    exp_results = pd.DataFrame(exp_results)
    out_fn = os.path.join(results_dir, f'{args.split}_{decode_suffix}{chunk_suffix}_ranges.csv')
    if not args.do_not_save:
        print(out_fn)
        exp_results.to_csv(out_fn, index=False)
    for col in list(exp_results.columns):
        print(f'{col}: min={exp_results[col].min()}, max={exp_results[col].max()}, avg={exp_results[col].mean()}')
