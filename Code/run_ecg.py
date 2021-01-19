"""
Usage:
    run.py --tsc (--dataset=<ds>) (--cv [--n_iter=<n>] [--score_function=<f>]| --default_split)
    run.py --etsc (--dataset=<ds>) (--cv [--n_iter=<n>] [--score_function=<f>]| --default_split) [--from_beg | --from_end] [--split=<s>]

Options:
    --tsc                  Runs an experiment on the dataset; to recommend the best performing model based on accuracy
    --etsc                 Runs an experiment on the dataset; to study performance of time series classification algorithms in an early classification context
    --dataset=<ds>         The name of the dataset folder. It should contain arff files, with the naming convention dataset_TEST.arff and dataset_TRAIN.arff
    --cv                   Apply cross validation to the dataset
    --default_split        Use the default split of the dataset
    --n_iter=<n>           Number of iterations for randomized cross validation [default: 50]
    --score_function=<f>   Function used for classification performance. Use a value from sklearn.metrics [default: balanced_accuracy]
    --from_beg             Start from beginning of time series and reveal next subsequences at each iteration
    --from_end             Start from end of time series and reveal previous subsequences at each iteration
    --split=<s>            The number of splits to apply to the time series. 10 splits= 10% increments, 20 splits= 5% increments,...etc [default: 20]
"""
from re import split
import docopt
import numpy as np
import pandas as pd
import logging
import os
from pyts import datasets as ds

if __name__ == "__main__":
        archive= ds.ucr_dataset_info()
        archive.update(ds.uea_dataset_info())

        archive_df= pd.DataFrame.from_dict(archive,orient='index')

        import IPython
        IPython.embed()
        