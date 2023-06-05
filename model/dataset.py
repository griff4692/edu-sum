import os

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

from model.data_utils import Seq2SeqCollate
from sum_constants import summarization_name_mapping


def remove_non_oracle(input_ids, oracle_labels, special_token_ids):
    s, e = special_token_ids
    start_locs = np.where(np.array(input_ids) == s)[0]
    end_locs = np.where(np.array(input_ids) == e)[0]

    n = len(start_locs)
    remove_idxs = []
    for idx in range(n):
        if idx not in oracle_labels:
            remove_idxs += [start_locs[idx], end_locs[idx]]

    keep_idxs = np.sort(list(set(list(range(len(input_ids)))) - set(remove_idxs)))
    return [input_ids[i] for i in keep_idxs]


def corrupt_indicators(input_ids, oracle_idxs, special_token_ids, corrupt_strategy):
    s, e = special_token_ids
    start_locs = np.where(np.array(input_ids) == s)[0]
    end_locs = np.where(np.array(input_ids) == e)[0]

    n = len(start_locs)
    oracle_n = len(oracle_idxs)
    assert n == len(end_locs)

    non_oracle_idxs = [i for i in range(n) if i not in oracle_idxs]
    non_oracle_n = len(non_oracle_idxs)

    if corrupt_strategy == 'random':
        num_to_replace = min(non_oracle_n, oracle_n)
        idx_to_keep = np.sort(np.random.choice(non_oracle_idxs, size=(num_to_replace,), replace=False))
    else:
        assert corrupt_strategy == 'swap'
        idx_to_keep = oracle_idxs.copy()
        if non_oracle_n >= 1:
            other_sent = int(np.random.choice(non_oracle_idxs))
            idx_to_keep[np.random.randint(oracle_n)] = other_sent
            idx_to_keep = list(np.sort(idx_to_keep))
        else:
            idx_to_keep = idx_to_keep[:-1]
    return remove_non_oracle(input_ids, idx_to_keep, special_token_ids)


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        pegasus_suffix = '_pegasus' if 'pegasus' in args.hf_model else ''
        data_dir = os.path.join(args.data_dir, args.dataset + f'_edu_alignments{pegasus_suffix}')
        print(f'Loading data from {data_dir}')
        self.dataset = load_from_disk(data_dir)
        self.tokenizer = tokenizer
        self.num_workers = 0 if args.debug else 8

    def get_train_chunk(self, chunk, num_chunks, **dataloader_kwargs):
        split_dataset = self.dataset['train']
        n = len(split_dataset)

        all_idxs = list(range(n))
        chunk_idxs = np.array_split(all_idxs, num_chunks)[chunk]
        print(f'Using {len(chunk_idxs)} training examples set for chunk {chunk}/{num_chunks}')
        print(f'First {min(3, len(chunk_idxs))} idxs: {chunk_idxs[:min(3, len(chunk_idxs))]}')
        split_dataset = split_dataset.select(chunk_idxs)

        split_dataset_pl = SummarizationDataset(self.args, split_dataset, self.tokenizer, 'train')
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            split='train',
        )
        kwargs = {
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        kwargs.update(**dataloader_kwargs)
        return DataLoader(split_dataset_pl, **kwargs), chunk_idxs

    def get_split(self, split, max_examples=None, **dataloader_kwargs):
        split_dataset = self.dataset[split]
        if self.args.debug and max_examples is None:
            max_examples = 128

        n = len(split_dataset)
        idxs = list(range(n))
        if max_examples is not None and max_examples < n:
            idxs = list(np.sort(np.random.choice(np.arange(n), size=(max_examples, ), replace=False)))
            print(f'First {min(10, len(idxs))} idxs sampled: {idxs[:min(10, len(idxs))]}')
            split_dataset = split_dataset.select(idxs)

        split_dataset_pl = SummarizationDataset(
            self.args, split_dataset, self.tokenizer, split
        )
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            max_input_length=self.args.max_input_length,
            split=split,
        )
        batch_size = self.args.per_device_train_bs if split == 'train' else self.args.per_device_eval_bs
        kwargs = {
            'batch_size': batch_size,
            'shuffle': split == 'train',
            'num_workers': self.num_workers,
            'collate_fn': collate_fn
        }
        kwargs.update(**dataloader_kwargs)
        return DataLoader(split_dataset_pl, **kwargs), idxs

    def train_dataloader(self, max_examples=None):
        return self.get_split('train', max_examples=None)[0]

    def val_dataloader(self, max_examples=None):
        return self.get_split('validation', max_examples=max_examples or self.args.max_val_examples)[0]

    def test_dataloader(self, max_examples=None):
        return self.get_split('test', max_examples=max_examples)[0]


class SummarizationDataset(Dataset):
    def __init__(self, args, dataset, tokenizer, split):
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.split = split
        _, self.target_col = summarization_name_mapping[self.args.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        dataset_id = example['id']
        target = example[self.target_col]

        oracle_labels = np.sort(example['oracle_idxs'])
        oracle_soft_labels = example['oracle_soft_labels']
        assert len(example['oracle_soft_labels']) == len(
            [x for x in example['input_ids'] if x == self.tokenizer.additional_special_tokens_ids[0]]
        )
        # Make sure you use same sentence tokenizer as in extract_oracles.py (otherwise oracle idxs may not align)
        source_annotated = example['source_edu_annotated']
        input_ids = example['input_ids']
        corrupt_input_ids = None
        plan_input_ids = None
        if not self.args.add_sent_toks:
            input_ids = [x for x in input_ids if x not in self.tokenizer.additional_special_tokens_ids]
        elif self.args.extract_indicators:
            # Remove Non-Oracle Markers
            corrupt_input_ids = corrupt_indicators(
                input_ids.copy(), oracle_labels.copy(), self.tokenizer.additional_special_tokens_ids,
                self.args.corrupt_strategy
            )
            plan_input_ids = remove_non_oracle(
                input_ids.copy(), oracle_labels.copy(), self.tokenizer.additional_special_tokens_ids
            )

            input_ids = [x for x in input_ids if x not in self.tokenizer.additional_special_tokens_ids]

        row = {
            'input_ids': input_ids,
            'labels': example['labels'],
            'source': source_annotated,
            'oracle_labels': oracle_labels,
            'oracle_soft_labels': oracle_soft_labels,
            'reference': target,  # Use for evaluation
        }

        if corrupt_input_ids is not None:
            row['corrupt_input_ids'] = corrupt_input_ids
            row['plan_input_ids'] = plan_input_ids

        return row
