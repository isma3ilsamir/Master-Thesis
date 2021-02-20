import numpy as np
import pandas as pd

def get_split_indexes(df, div, dataset):
    print(f"Applying splitting for dataset: {dataset}")
    ts_length= len(df['dim_0'][0])
    split_lengths= ([ts_length // div + (1 if x < ts_length % div else 0)  for x in range (div)])
    split_indexes= np.cumsum(split_lengths).tolist()
    split_indexes= keep_chunks(split_indexes, [1,2,3,len(split_indexes)])
    return split_indexes

def get_split_indexes_w_threshold(df, div, threshold, dataset):
    ts_length= len(df['dim_0'][0])
    ts_length_10pct = int(ts_length * 0.1)
    if (ts_length < threshold) | (ts_length_10pct >= threshold) :
        split_lengths= ([ts_length // div + (1 if x < ts_length % div else 0)  for x in range (div)])
        split_indexes= np.cumsum(split_lengths).tolist()
    elif ts_length_10pct < threshold :
        quot , rem = divmod(ts_length, threshold)
        split_lengths= [threshold] * quot
        split_lengths.append(rem)
        split_indexes= np.cumsum(split_lengths).tolist()
    else:
        print("################ !!!! WARNING !!!! ################")
        print(f"Dividing using threshold skipped both conditions {dataset}")
    if len(split_indexes) > 4:
        split_indexes= keep_chunks(split_indexes, [1,2,3,len(split_indexes)])
    return split_indexes

def apply_split(df,split_index, asc=True):
    if not asc:
        df = df.applymap(lambda cell: cell[-split_index:])
        df = df.applymap(lambda cell: cell.reset_index(drop=True))
    else:
        df = df.applymap(lambda cell: cell[:split_index])
    return df

def keep_chunks(split_indexes, chunks):
    remaining_indexes= [split_indexes[i - 1] for i in chunks]
    return remaining_indexes

