"""
Usage:
    run.py --dataset=<ds>

Options:
    --dataset=<ds>         The name of the dataset folder. It should contain arff files, with the naming convention dataset_TEST.arff and dataset_TRAIN.arff
"""
import sys
from os import path
# Add parent import capabilities
MAIN_DIRECTORY = path.dirname(path.dirname(path.realpath(__file__)))
if MAIN_DIRECTORY not in sys.path:
    sys.path.insert(0, MAIN_DIRECTORY)

import docopt
import pandas as pd
import logging
import concurrent.futures
from datasets import get_test_train_data
import numpy as np

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
def main():
    arguments=docopt.docopt(__doc__)
    if isinstance(arguments['--dataset'], str):
        arguments['--dataset'] = [arguments['--dataset']]

    try:
        # download the datasets
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # executor.map(get_test_train_data, arguments['--dataset'])
            futures = {
                executor.submit(get_test_train_data, task) : task
                    for task in arguments['--dataset']
            }
            concurrent.futures.wait(futures)
    except Exception as e:
        logger.info(f"Exception occurred: {e}")

    records=[]
    for fut in concurrent.futures.as_completed(futures):
        _, _, y_train, _ = fut.result()
        (unique, counts) = np.unique(y_train, return_counts=True)
        mx= counts.max()
        mn= counts.min()
        balanced = False if mx >= (2 * mn) else True
        record= {
                    'dataset' : futures[fut],
                    'balanced' : balanced,
                    'counts' : counts
                }
        records.append(record)
        # frequencies = np.asarray((unique, counts)).T

    df= pd.DataFrame(records)
    import IPython
    IPython.embed()

if __name__ == "__main__":
    main()