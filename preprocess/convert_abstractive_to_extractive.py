import gc
import glob
import gzip
import itertools
import json
import logging
import math
import os
import re
import sys
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from time import time

import pandas as pd
import spacy
import numpy as np
from spacy.lang.en import English
from tqdm import tqdm

import datasets as hf_nlp
from preprocess.helpers import _get_word_ngrams, load_json
from sum_constants import summarization_name_mapping


logger = logging.getLogger(__name__)

# Steps
# Run cnn/dm processing script to get train, test, valid bin text files
# For each bin:
#   Load all the data where each line is an entry in a list
#   For each document (line) (parallelized):
#       Tokenize line in the source and target files
#       Run source and target through oracle_id algorithm
#       Run current preprocess_examples() function (data cleaning) in data processor
#       Return source (as list of sentences) and target
#   In map() loop: append each (source, target, labels) to variable and save (as
#   cnn_dm_extractive) once done

# BertSum:
# 1. Tokenize all files into tokenized json versions
# 2. Split json into source and target AND concat stories into chunks of `shard_size`
#    number of stories
# 3. Process to obtain extractive summary and labels for each shard
# 4. Save each processed shard as list of dictionaries with processed values


def read_in_chunks(file_object, chunk_size=5000):
    """Read a file line by line but yield chunks of ``chunk_size`` number of lines at a time."""
    # https://stackoverflow.com/a/519653
    # zero mod anything is zero so start counting at 1
    current_line_num = 1
    lines = []
    for line in file_object:
        # use `(chunk_size + 1)` because each iteration starts at 1
        if current_line_num % (chunk_size + 1) == 0:
            yield lines
            # reset the `lines` so a new chunk can be yielded
            lines.clear()
            # Essentially adds one twice so that each interation starts counting at one.
            # This means each yielded chunk will be the same size instead of the first
            # one being 5000 then every one after being 5001, for example.
            current_line_num += 1
        lines.append(line.strip())
        current_line_num += 1
    # When the `file_object` has no more lines left then yield the current chunk,
    # even if it is not a chunk of 5000 (`chunk_size`) as long as it contains more than
    # 0 examples.
    if len(lines) > 0:
        yield lines


def resume(output_path, split, chunk_size):
    """
    Find the last shard created and return the total number of lines read and last
    shard number.
    """
    glob_str = os.path.join(output_path, (split + ".*.json*"))
    all_json_in_split = glob.glob(glob_str)

    if not all_json_in_split:  # if no files found
        return None

    # get the first match because and convert to int so max() operator works
    # more info about the below RegEx: https://stackoverflow.com/a/1454936
    # (https://web.archive.org/web/20200701145857/https://stackoverflow.com/questions/1454913/regular-expression-to-find-a-string-included-between-two-characters-while-exclud/1454936) # noqa: E501
    shard_file_idxs = [
        int(re.search(r"(?<=\.)(.*?)(?=\.)", a).group(1)) for a in all_json_in_split
    ]

    last_shard = int(max(shard_file_idxs)) + 1  # because the file indexes start at 0

    num_lines_read = chunk_size * last_shard
    # `num_lines_read` is the number of lines read if line indexing started at 1
    # therefore, this number is the number of the next line wanted
    return num_lines_read, last_shard


def check_resume_success(
    nlp, args, source_file, last_shard, output_path, split, compression
):
    logger.info("Checking if resume was successful...")
    chunk_file_path_str = split + "." + str(last_shard - 1) + ".json"
    if compression:
        chunk_file_path_str += ".gz"
    chunk_file_path = os.path.join(output_path, chunk_file_path_str)

    line_source = source_file.readline().strip()

    line_source_tokenized = next(tokenize(nlp, [line_source]))

    # Apply preprocessing on the line
    preprocessed_line = preprocess(
        line_source_tokenized,
        [1] * len(line_source_tokenized),
        args.min_sentence_ntokens,
        args.max_sentence_ntokens,
        args.min_example_nsents,
        args.max_example_nsents,
    )[0]

    try:
        chunk_json, _ = load_json(chunk_file_path)
    except FileNotFoundError:
        logger.error(
            "The file at path %s was not found. Make sure `--compression` is set correctly.",
            chunk_file_path,
        )
    last_item_chunk = chunk_json[-1]
    line_chunk = last_item_chunk["src"]

    # remove the last item if it is a newline
    if line_chunk[-1] == ["\n"]:
        line_chunk.pop()

    if line_chunk == preprocessed_line:
        logger.info("Resume Successful!")
        logger.debug("`source_file` moved forward one line")
    else:
        logger.info("Resume NOT Successful")
        logger.info("Last Chunk Line: %s", line_chunk)
        logger.info("Previous (to resume line) Source Line: %s", preprocessed_line)
        # skipcq: PYL-W1201
        logger.info(
            "Common causes of this issue:\n"
            + "1. You changed the `--shard_interval`. You used a different interval previously "
            + "than you used in the command to resume.\n"
            + "2. The abstractive (`.source` and `.target`) or extractive (`.json`) dataset "
            + "files were modified or removed. The last `.json` file needs to be in the same "
            + "folder it was originally outputted to so the last shard index and be determined "
            + "and the last line can be read.\n"
            + "3. It is entirely possible that there is a bug in this script. If you have checked "
            + "that the above were not the cause and that there were no issues pertaining to your "
            + "dataset then open an issue at https://github.com/HHousen/TransformerSum/issues/new."
        )
        return False

    return True


def seek_files(files, line_num):
    """Seek a set of files to line number ``line_num`` and return the files."""
    rtn_file_objects = []
    for file_object in files:
        offset = 0
        for idx, line in enumerate(file_object):
            if idx >= line_num:
                break
            offset += len(line)
        file_object.seek(0)

        file_object.seek(offset)
        rtn_file_objects.append(file_object)
    return rtn_file_objects


def save(json_to_save, output_path, compression=False):
    """
    Save ``json_to_save`` to ``output_path`` with optional gzip compresssion
    specified by ``compression``.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving to %s", output_path)
    if compression:
        # https://stackoverflow.com/a/39451012
        json_str = json.dumps(json_to_save)
        json_bytes = json_str.encode('utf-8')
        with gzip.open((output_path + '.gz'), 'w') as save_file:
            save_file.write(json_bytes)
    else:
        with open(output_path, 'w') as save_file:
            save_file.write(json.dumps(json_to_save))


def tokenize_for_oracle(
        nlp,
        docs,
        n_process=5,
        batch_size=100,
        name="",
        tokenizer_log_interval=0.1,
        disable_progress_bar=False,
):
    """Tokenize using spacy and split into sentences and tokens."""
    tokenized = []
    for doc in tqdm(
            nlp.pipe(
                docs,
                n_process=n_process,
                batch_size=batch_size,
            ),
            total=len(docs),
            desc='Tokenizing' + name,
            mininterval=tokenizer_log_interval,
            disable=disable_progress_bar,
    ):
        tokenized.append(doc)
    return [list(doc.sents) for doc in tokenized]


def tokenize(
    nlp,
    docs,
    n_process=5,
    batch_size=100,
    name="",
    tokenizer_log_interval=0.1,
    disable_progress_bar=False,
    return_str=False
):
    """Tokenize using spacy and split into sentences and tokens."""
    tokenized = []

    for doc in tqdm(
        nlp.pipe(
            docs,
            n_process=n_process,
            batch_size=batch_size,
        ),
        total=len(docs),
        desc='Tokenizing' + name,
        mininterval=tokenizer_log_interval,
        disable=disable_progress_bar,
    ):
        tokenized.append(doc)

    logger.debug('Splitting into sentences and tokens and converting to lists')
    t0 = time()

    doc_sents = (doc.sents for doc in tokenized)

    if return_str:
        del doc_sents

    del tokenized
    del docs
    sents = (
        [[token.text for token in sentence] for sentence in doc] for doc in doc_sents
    )

    logger.debug('Done in %.2f seconds', time() - t0)
    # `sents` is an array of documents where each document is an array sentences where each
    # sentence is an array of tokens
    return sents


def example_processor(inputs, args, oracle_mode="greedy", no_preprocess=False):
    """
    Create ``oracle_ids``, convert them to ``labels`` and run
    :meth:`~convert_to_extractive.preprocess`.
    """
    source_doc, target_doc, prediction_doc = inputs
    selection = None
    if oracle_mode == 'greedy':
        oracle_ids, oracle_rouges = greedy_selection(source_doc, target_doc, 5)
        pred_oracle_ids, pred_rouges = greedy_selection(source_doc, prediction_doc, 5)
    elif oracle_mode == 'combination':
        oracle_ids, oracle_rouges = combination_selection(source_doc, target_doc, 3)
        pred_oracle_ids, pred_rouges = combination_selection(source_doc, prediction_doc, 3)
    elif oracle_mode == 'oracle':
        oracle_combo_ids, oracle_combo_rouges = combination_selection(source_doc, target_doc, 3)
        pred_combo_ids, pred_combo_rouges = combination_selection(source_doc, prediction_doc, 3)

        oracle_gain_ids, oracle_gain_rouges, _, _, _ = gain_selection(source_doc, target_doc, 5)
        pred_gain_ids, pred_gain_rouges, _, _, _ = gain_selection(source_doc, prediction_doc, 5)

        if sum(oracle_combo_rouges.values()) > sum(oracle_gain_rouges.values()):
            oracle_ids, oracle_rouges = oracle_combo_ids, oracle_combo_rouges
        else:
            oracle_ids, oracle_rouges = oracle_gain_ids, oracle_gain_rouges

        if sum(pred_combo_rouges.values()) > sum(pred_gain_rouges.values()):
            pred_oracle_ids, pred_oracle_rouges = pred_combo_ids, pred_combo_rouges
            selection = 'combination'
        else:
            pred_oracle_ids, pred_oracle_rouges = pred_gain_ids, pred_gain_rouges
            selection = 'gain'
        if sum(pred_combo_rouges.values()) == sum(pred_gain_rouges.values()):
            selection = 'tie'
    else:
        oracle_ids, oracle_rouges, _, _, _ = gain_selection(source_doc, target_doc, 5)
        pred_oracle_ids, pred_rouges, _, _, _ = gain_selection(source_doc, prediction_doc, 5)
    oracle_sents = [source_doc[i] for i in oracle_ids]
    oracle_results = _calc_rouge(target_doc, oracle_sents)
    pred_sents = [source_doc[i] for i in pred_oracle_ids]
    abs_results = _calc_rouge(target_doc, pred_sents)

    # compute ROUGE of oracle extractive upper bound against predicted summary
    upper_score = _calc_rouge(prediction_doc, oracle_sents)
    pred_score = _calc_rouge(prediction_doc, pred_sents)

    r1_gap = pred_score['rouge_1'] - upper_score['rouge_1']
    r2_gap = pred_score['rouge_2'] - upper_score['rouge_2']

    rand_rank = _calc_rouge(prediction_doc, list(np.random.choice(source_doc, size=(len(oracle_ids)), replace=False)))

    # print('Abs: ', ' '.join([str(x) for x in abs_results.values()]))
    # print('Oracle: ', ' '.join([str(x) for x in oracle_results.values()]))

    results = {
        'oracle_rouge_1': oracle_results['rouge_1'],
        'oracle_rouge_2': oracle_results['rouge_2'],
        'abs_rouge_1': abs_results['rouge_1'],
        'abs_rouge_2': abs_results['rouge_2'],
        'pred_rouge_1': pred_score['rouge_1'],
        'pred_rouge_2': pred_score['rouge_2'],
        'upper_rouge_1': upper_score['rouge_1'],
        'upper_rouge_2': upper_score['rouge_2'],
        'rand_rouge_1': rand_rank['rouge_1'],
        'rand_rouge_2': rand_rank['rouge_2'],
        'gap_rouge_1': r1_gap,
        'gap_rouge_2': r2_gap,
        'selection': selection
    }
    # `oracle_ids` to labels
    labels = [0] * len(source_doc)
    for l_id in oracle_ids:
        labels[l_id] = 1

    # The number of sentences in the source document should equal the number of labels.
    # There should be one label per sentence.
    assert len(source_doc) == len(labels), (
        "Document: "
        + str(source_doc)
        + "\nLabels: "
        + str(labels)
        + "\n^^ The above document and label combination are not equal in length. The cause of "
        + "this problem in not known. This check exists to prevent further problems down the "
        + "data processing pipeline."
    )

    if no_preprocess:
        preprocessed_data = source_doc, labels
    else:
        preprocessed_data = preprocess(
            source_doc,
            labels,
            args.min_sentence_ntokens,
            args.max_sentence_ntokens,
            args.min_example_nsents,
            args.max_example_nsents,
        )

    return preprocessed_data, target_doc, results


def preprocess(
    example,
    labels,
    min_sentence_ntokens=5,
    max_sentence_ntokens=200,
    min_example_nsents=3,
    max_example_nsents=100,
):
    """
    Removes sentences that are too long/short and examples that have
    too few/many sentences.
    """
    # pick the sentence indexes in `example` if they are larger then `min_sentence_ntokens`
    idxs = [i for i, s in enumerate(example) if (len(s) > min_sentence_ntokens)]
    # truncate selected source sentences to `max_sentence_ntokens`
    example = [example[i][:max_sentence_ntokens] for i in idxs]
    # only pick labels for sentences that matched the length requirement
    labels = [labels[i] for i in idxs]
    # truncate entire source to max number of sentences (`max_example_nsents`)
    example = example[:max_example_nsents]
    # perform above truncation to `labels`
    labels = labels[:max_example_nsents]

    # if the example does not meet the length requirement then return None
    if len(example) < min_example_nsents:
        return None
    return example, labels


def _calc_rouge(ref_sents, pred_sents):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    ref_sents = sum(ref_sents, [])
    pred_sents = sum(pred_sents, [])
    abstract = _rouge_clean(" ".join(ref_sents)).split()
    pred = _rouge_clean(" ".join(pred_sents)).split()
    evaluated_1grams = _get_word_ngrams(1, [pred])
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = _get_word_ngrams(2, [pred])
    reference_2grams = _get_word_ngrams(2, [abstract])
    rouge_1 = cal_rouge(evaluated_1grams, reference_1grams)['f']
    rouge_2 = cal_rouge(evaluated_2grams, reference_2grams)['f']
    return {
        'rouge_1': rouge_1,
        'rouge_2': rouge_2
    }


# Section Methods (to convert abstractive summary to extractive)
# Copied from https://github.com/nlpyang/BertSum/blob/9aa6ab84faf3a50724ce7112c780a4651de289b0/src/prepro/data_builder.py  # noqa: E501
def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    best_rouges = {}
    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
                best_rouges['rouge_1'] = rouge_1
                best_rouges['rouge_2'] = rouge_2
    return sorted(list(max_idx)), best_rouges


def gain_selection(doc_sent_list, abstract_sent_list, summary_size, lower=False, sort=False):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    curr_summary_1grams = set()
    curr_summary_2grams = set()
    abstract = sum(abstract_sent_list, [])
    if lower:
        abstract = _rouge_clean(' '.join(abstract)).split()
        sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    else:
        abstract = _rouge_clean(' '.join(abstract).lower()).split()
        sents = [_rouge_clean(' '.join(s).lower()).split() for s in doc_sent_list]
    n = len(sents)
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    best_rouges = {'rouge_1': -1, 'rouge_2': -1}
    # impossible_sents = []
    max_idxs = []
    r1_history = []
    r2_history = []
    best_history = []
    for s in range(summary_size + 1):
        best_rouge_1, best_rouge_2, best_idx = -1, -1, -1
        row_r1 = []
        row_r2 = []
        for source_idx in range(n):
            # if source_idx in impossible_sents:
            #     continue

            candidates_1 = evaluated_1grams[source_idx].union(curr_summary_1grams)
            candidates_2 = evaluated_2grams[source_idx].union(curr_summary_2grams)
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            row_r1.append(str(rouge_1))
            row_r2.append(str(rouge_2))

            # if s == 0 and rouge_1 + rouge_2 == 0:
            #     impossible_sents.append(source_idx)
            if rouge_1 + rouge_2 > best_rouge_1 + best_rouge_2:
                best_rouge_1 = rouge_1
                best_rouge_2 = rouge_2
                best_idx = source_idx
        if best_rouge_1 + best_rouge_2 > best_rouges['rouge_1'] + best_rouges['rouge_2']:
            best_rouges['rouge_1'] = best_rouge_1
            best_rouges['rouge_2'] = best_rouge_2
            best_history.append(f'{best_rouge_1},{best_rouge_2}')
            max_idxs.append(best_idx)
            curr_summary_1grams = curr_summary_1grams.union(evaluated_1grams[best_idx])
            curr_summary_2grams = curr_summary_2grams.union(evaluated_2grams[best_idx])

            r1_history.append(','.join(row_r1))
            r2_history.append(','.join(row_r2))
        else:
            break
    if sort:
        max_idxs = list(sorted(max_idxs))
    return max_idxs, best_rouges, '|'.join(r1_history), '|'.join(r2_history), '|'.join(best_history)


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    best_rouges = {'rouge_1': 0, 'rouge_2': 0}

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
                best_rouges['rouge_1'] = rouge_1
                best_rouges['rouge_2'] = rouge_2
        if cur_id == -1:
            return selected, best_rouges
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), best_rouges


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {'f': f1_score, 'p': precision, 'r': recall}
