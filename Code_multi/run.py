"""
Usage:
    run.py --tsc (--dataset=<ds>)... (--cv [--n_iter=<n>] [--score_function=<f>]| --default_split)
    run.py --etsc (--dataset=<ds>)... (--cv [--n_iter=<n>] [--score_function=<f>]| --default_split) [--from_beg | --from_end] [--split=<s>]

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
import docopt
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import concurrent.futures
from itertools import repeat

from splitting import get_split_indexes, apply_split
from datasets import get_test_train_data

import matplotlib.pyplot as plt

from models import KNNED, KNNDTW, KNNMSM, KNNED_SKTIME, PFOREST, EE
from models import KNNDTW_sc, KNNDTW_it, KNNDTW_ms, KNNDTW_fs
from models import TSF
from models import LS, ST
from models import WEASEL, CBOSS
from models import INCEPTION


def initialize_logger():
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

    return logger


def tsc(logger, args, X_train, X_test, y_train, y_test):
    logger.info(
        f"===== Starting Time Series Classification Model Selection Experiment =====")

    logger.info(f"===== Step: Initializaing models =====")
    models = {
        'knn_ed': KNNED(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # # 'knn_ed_sktime': KNNED_SKTIME(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'knn_dtw': KNNDTW(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'knn_msm': KNNMSM(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # # 'ee': EE(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'tsf': TSF(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'ls': LS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'st': ST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'weasel': WEASEL(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'cboss': CBOSS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'inception': INCEPTION(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'pforest': PFOREST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'knn_dtw_sc': KNNDTW_sc(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_it': KNNDTW_it(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_ms': KNNDTW_ms(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_fs': KNNDTW_fs(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset'])
    }

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(tsc_process_model,
                     repeat(logger),
                     repeat('tsc'),
                     models.values(),
                     repeat(X_train),
                     repeat(X_test),
                     repeat(y_train),
                     repeat(y_test)
                )

def tsc_process_model(logger, process, m,  X_train, X_test, y_train, y_test):
    analysis_data = []
    if args['cv']:
        analysis_data.append(cv(logger, process, m, X_train, X_test, y_train, y_test, None))
    elif args['default_split']:
        analysis_data.append(no_cv(logger, process, m, X_train, X_test, y_train, y_test, None))

    analysis = pd.DataFrame(analysis_data)
    analysis['dataset'] = args['dataset']

    logger.info(f"===== Step: Exporting Analysis Results for {m.clf_name} =====")
    export_analysis(analysis, args)

def etsc(logger, args, X_train, X_test, y_train, y_test):
    logger.info(
        f"===== Starting Early Time Series Classification Experiment =====")

    logger.info(f"===== Step: Initializaing models =====")
    models = {
        # 'knn_ed': KNNED(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # # 'knn_ed_sktime': KNNED_SKTIME(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'knn_dtw': KNNDTW(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'knn_msm': KNNMSM(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # #   'ee': EE(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'tsf': TSF(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'ls': LS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'st': ST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'weasel': WEASEL(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'cboss': CBOSS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'inception': INCEPTION(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        'pforest': PFOREST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'knn_dtw_sc': KNNDTW_sc(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_it': KNNDTW_it(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_ms': KNNDTW_ms(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_dtw_fs': KNNDTW_fs(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset'])
    }

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(etsc_process_model,
                     repeat(logger),
                     repeat('etsc'),
                     models.values(),
                     repeat(X_train),
                     repeat(X_test),
                     repeat(y_train),
                     repeat(y_test)
                )

def etsc_process_model(logger, process, m, X_train, X_test, y_train, y_test):
    dfs = []
    analysis_data = []
    logger.info(f"===== Step: Applying data splitting =====")
    split_indexes = get_split_indexes(X_train, args['split'])
    logger.info(
        f"Data split Indexes obtained. Time series length will be split into {args['split']} approximaltely equal chunks")

    for i, index in enumerate(split_indexes):
        revealed_pct= (100 / args["split"])*(i+1)
        logger.info(f'######################################################')
        logger.info(
            f'===== Learning on {revealed_pct}% of the time series =====')
        X_train_splitted = apply_split(
            X_train, index, desc=arguments['--from_beg'])

        if args['cv']:
            analysis_data.append(cv(logger, process, m, X_train_splitted, X_test,
                            y_train, y_test, revealed_pct))

        elif args['default_split']:
            analysis_data.append(no_cv(logger, process, m, X_train_splitted, X_test,
                                y_train, y_test, revealed_pct))

        df = pd.DataFrame(analysis_data)
        df['revealed_pct'] = revealed_pct
        df['harmonic_mean'] = (2 * (1 - df['revealed_pct']) * df['test_ds_score']
                            )/((1 - df['revealed_pct']) + df['test_ds_score'])
        dfs.append(df)
    analysis = pd.concat(dfs, axis=0, ignore_index=True)
    analysis['dataset'] = args['dataset']
    logger.info(f"===== Step: Exporting Analysis Results for {m.clf_name} =====")
    export_analysis(analysis, args)

def cv(logger, process, model, X_train, X_test, y_train, y_test, revealed_pct):
    logger.info(f"===== Step: Running Cross Validation for {model.clf_name} =====")
    model.fit(X_train, y_train)
    model.best_estimator_analysis['train_time'] = model.train_time

    logger.info(f"===== Step: Labeling Testing Dataset for {model.clf_name} =====")
    model.predict(X_test)

    logger.info(f"===== Step: Calculating Accuracy scores for {model.clf_name} =====")
    model.get_score(X_test, y_test)
    model.best_estimator_analysis['test_ds_score'] = model.test_ds_score
    model.best_estimator_analysis['test_ds_score_time'] = model.test_ds_score_time

    logger.info(f"===== Step: Prepare Analysis Results for {model.clf_name} =====")
    analysis= model.best_estimator_analysis
    logger.info(f"Analysis metrics: score, fitting time, testing time")
    # logger.info("models are ranked based on each of the metrics")

    logger.info(f"===== Step: Exporting fitted model for {model.clf_name} =====")
    export_model(model, process, revealed_pct)
    return analysis


def no_cv(logger, process, model, X_train, X_test, y_train, y_test, revealed_pct):
    logger.info(f"===== Step: Fitting models on Training Dataset for {model.clf_name} =====")
    model.fit(X_train, y_train)

    logger.info(f"===== Step: Labeling Testing Dataset for {model.clf_name} =====")
    model.predict(X_test)

    logger.info(f"===== Step: Calculating Accuracy scores for {model.clf_name} =====")
    model.get_score(X_test, y_test)

    logger.info(f"===== Step: Prepare Analysis Results for {model.clf_name} =====")
    idx = ['classifier', 'train_time','test_ds_score_time', 'test_ds_score', 'params']
    analysis= pd.Series([model.clf_name, model.train_time, model.test_ds_score_time, model.test_ds_score, model.clf.get_params()], index=idx)
    logger.info("Analysis metrics: score, fitting time, testing time")
    # logger.info("models are ranked based on each of the metrics")

    logger.info(f"===== Step: Exporting fitted model for {model.clf_name} =====")
    export_model(model, process, revealed_pct)
    return analysis


def get_analysis_file_name(args):
    analysis_folder = os.path.join(os.getcwd(), 'analysis','')
    process = 'tsc' if args['tsc'] else 'etsc'
    cv = 'cv' if args['cv'] else 'no_cv'
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    fname = os.path.join(analysis_folder, f"{args['dataset']}_{process}_{cv}")
    return fname


def get_model_analysis_file_name(process, dataset, cv, split, model_name):
    analysis_folder = os.path.join(
        os.getcwd(), 'datasets', dataset, 'analysis','')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    f = f"{process}_{model_name}_{cv}_{split}" if split else f"{process}_{model_name}_{cv}"
    fname = os.path.join(
        analysis_folder, f)
    return fname


def export_analysis(analysis, args):
    process = 'tsc' if args['tsc'] else 'etsc'
    cv = 'cv' if args['cv'] else 'no_cv'
    dataset = args['dataset']
    split= f"{args['split']}split" if args['split'] else None

    clf_list = analysis['classifier'].unique().tolist()
    clf_groups = analysis.groupby(analysis.classifier)

    for clf in clf_list:
        clf_grouped = clf_groups.get_group(clf)
        model_analysis_file=get_model_analysis_file_name(process, dataset, cv, split, clf)
        clf_grouped.to_json(f'{model_analysis_file}.json')
        logger.info(
            f"Analysis for {clf} exported to {model_analysis_file}.json")


def export_model(model, process, revealed_pct=None):
    if model.clf == 'PForest':
        logger.critical('Proximity Forest Model export is not currently working')
        pass
    model.export_model(process,revealed_pct)

def get_report_file_name(start_time):
    ts= datetime.strftime(start_time,"%Y_%m_%d_%H%M%S")
    report_folder = os.path.join(
        os.getcwd(), 'reports', '')
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)
    f = f"{__file__}_{ts}"
    fname = os.path.join(
        report_folder, f)
    return fname

if __name__ == "__main__":
    start_time= datetime.now()
    report_ts= datetime.strftime(start_time,"%d-%m-%Y %H:%M:%S")
    report= []
    analysis=None
    models=None
    logger=initialize_logger()
    arguments=docopt.docopt(__doc__)
    for dataset in arguments['--dataset']:
        try:
            args=dict()
            args['tsc']=arguments['--tsc']
            args['etsc']=arguments['--etsc']
            args['cv']=arguments['--cv']
            args['default_split']=arguments['--default_split']
            args['dataset']=dataset
            args['n_iter']=int(arguments['--n_iter'])
            args['score_function']=arguments['--score_function']
            args['split']= None if arguments['--tsc'] else int(arguments['--split'])
            if arguments['--from_beg'] | (not arguments['--from_beg'] and not arguments['--from_end']):
                args['from_beg']=True
            args['from_end']=arguments['--from_end']

            logger.info(
                f"===== Step: Loading Dataset {args['dataset']} =====")
            X_train, X_test, y_train, y_test=get_test_train_data(
                args['dataset'])
            logger.info(f"Dataset {args['dataset']} loaded !!")
            args['dim']=X_train.shape[1]
            if args['dim'] > 1:
                logger.info(f"Dataset {args['dataset']} is Multivariate")
            else:
                logger.info(f"Dataset {args['dataset']} is Univariate")

            if args['tsc']:
                tsc(logger, args, X_train, X_test, y_train, y_test)
            elif args['etsc']:
                etsc(logger, args, X_train, X_test, y_train, y_test)
            
            report.append({'running_file':__file__, 'ts': start_time, 'tsc': args['tsc'], 'etsc': args['etsc'], 'dataset': args['dataset'], 'cv': args['cv'], 'default_split': args['default_split'], 'n_iter': args['n_iter'], 'score_function': args['score_function'], 'split': args['split'], 'from_beg': args['from_beg'], 'from_end': args['from_end'], 'split': args['split'], 'success': True, 'exception':None})

        except Exception as e:
            report.append({'running_file':__file__, 'ts': start_time, 'tsc': args['tsc'], 'etsc': args['etsc'], 'dataset': args['dataset'], 'cv': args['cv'], 'default_split': args['default_split'], 'n_iter': args['n_iter'], 'score_function': args['score_function'], 'split': args['split'], 'from_beg': args['from_beg'], 'from_end': args['from_end'], 'split': args['split'], 'success': False, 'exception':e})
            logger.critical(e)
    report_df= pd.DataFrame(report)
    fname = get_report_file_name(start_time)
    report_df.to_json(f'{fname}.json',orient='records')
    logger.info(f"Running Report exported to {fname}.json")
