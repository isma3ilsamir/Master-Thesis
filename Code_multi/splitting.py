import numpy as np
import pandas as pd

def split_ts_5_pct(df, desc=False):
    df, split_indexes = get_splitted_data(df, desc)
    return df, split_indexes

def get_split_indexes(df, div=20):
    ts_length= len(df['dim_0'][0])
    # # div=20
    # div= 5
    split_lengths= ([ts_length // div + (1 if x < ts_length % div else 0)  for x in range (div)])
    split_indexes= np.cumsum(split_lengths)
    return split_indexes

def apply_split(df,split_index, desc=False):
    if desc:
        df = df.applymap(lambda cell: cell[-split_index:])
    else:
        df = df.applymap(lambda cell: cell[:split_index])
    return df

def get_splitted_data(df, desc=False):
    dfs = []
    split_indexes = get_split_indexes(df)
    for i in range(len(split_indexes)):
        splitted_df = apply_split(df, split_indexes[i], desc)
        splitted_df = splitted_df.add_prefix(f'{5*(i+1)}_pct_')
        dfs.append(splitted_df)
    df = pd.concat(dfs, axis=1)
    return df, split_indexes
