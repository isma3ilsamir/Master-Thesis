import math
import numpy as np
import pandas as pd

def split_ts(df):
    df, splits = _get_splitted_data(df)
    return df, splits

def _divisorGenerator(ts_length):
    divisors = []
    for i in range(1, int(math.sqrt(ts_length) + 1)):
        if ts_length % i == 0:
            divisors.append(i)
    divisors = divisors[1:]
    return divisors

def _get_splits(ts_length):
    default_splits = [2, 3, 4, 5, 10]
    divisor_splits = _divisorGenerator(ts_length)
    if divisor_splits:
        return divisor_splits
    return default_splits

def _apply_split(df, split_size):
    df = df.applymap(lambda cell: np.array_split(cell, split_size)[0])
    df = df.add_prefix(f'{split_size}split')
    return df

def _get_splitted_data(df):
    dfs = []
    dfs.append(df)
    ts_full_length = len(df['dim_0'][0])
    splits = _get_splits(ts_full_length)
    for split in splits:
        splitted_df = _apply_split(df, split)
        dfs.append(splitted_df)
    df = pd.concat(dfs, axis=1)
    return df, splits