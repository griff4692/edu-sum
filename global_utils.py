import datetime
import subprocess
import unicodedata
import random

import pandas as pd
import numpy as np
import torch


def set_same_seed(seed):
    # Set same random seed for each run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set random, numpy and torch seeds to {seed}')


def counts_to_df(counter, k):
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df = df.rename(columns={'index': k, 0: 'count'})
    return df


def get_free_gpus():
    try:
        gpu_stats = subprocess.check_output(
            ['nvidia-smi', '--format=csv,noheader', '--query-gpu=memory.used'], encoding='UTF-8')
        used = list(filter(lambda x: len(x) > 0, gpu_stats.split('\n')))
        return [idx for idx, x in enumerate(used) if int(x.strip().rstrip(' [MiB]')) <= 500]
    except:
        return []


def counter_to_pd(counter, key, ascending=False):
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df.rename(columns={'index': key, 0: 'count'}, inplace=True)
    df.sort_values(by='count', ascending=ascending, inplace=True)
    return df


def decode_utf8(str):
    """
    :param str: string with utf-8 characters
    :return: string with all ascii characters

    This is necessary for the ROUGE nlp package consumption.
    """
    return unicodedata.normalize(u'NFKD', str).encode('ascii', 'ignore').decode('utf8').strip()


def is_file_older_than(file, hours=3):
    delta = datetime.timedelta(hours=hours)
    cutoff = datetime.datetime.utcnow() - delta
    mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(file))
    if mtime < cutoff:
        return True
    return False


def tens_to_np(t):
    try:
        return t.numpy()
    except:
        return t.cpu().numpy()


def now_str():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
