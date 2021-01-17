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

from splitting import get_split_indexes, apply_split
from datasets import get_test_train_data

import matplotlib.pyplot as plt

from models import KNNED, KNNDTW, KNNMSM, KNNED_SKTIME, PFOREST, EE
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
    analysis_data = None
    logger.info(
        f"===== Starting Time Series Classification Model Selection Experiment =====")

    logger.info(f"===== Step: Initializaing models =====")
    models = {
        # 'knn_ed': KNNED(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_ed_sktime': KNNED_SKTIME(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'knn_dtw': KNNDTW(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'knn_msm': KNNMSM(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # #   'ee': EE(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'tsf': TSF(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # 'ls': LS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # # 'st': ST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        # 'weasel': WEASEL(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'cboss': CBOSS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'inception': INCEPTION(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset'])
        # 'pforest': PFOREST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset'])

    }

    if args['cv']:
        analysis_data = cv(logger, models, X_train, X_test,
                           y_train, y_test)
    elif args['default_split']:
        analysis_data = no_cv(logger, models, X_train, X_test,
                              y_train, y_test)
    
    logger.info(f"===== Step: Exporting fitted models =====")
    for m in models.values():
        export_model(m, 'tsc')

    analysis = pd.DataFrame(analysis_data)
    analysis['dataset'] = args['dataset']
    # analysis['rank_test_score'] = analysis['test_ds_score'].rank(
    #     ascending=False, method='dense')
    # analysis['rank_train_time'] = analysis['train_time'].rank(
    #     ascending=True, method='dense')
    # analysis['rank_score_time'] = analysis['test_ds_score_time'].rank(
    #     ascending=True, method='dense')

    return analysis, models


def etsc(logger, args, X_train, X_test, y_train, y_test):
    analysis_data = None
    logger.info(
        f"===== Starting Early Time Series Classification Experiment =====")

    logger.info(f"===== Step: Initializaing models =====")
    models = {
        # 'knn_ed': KNNED(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        #   'knn_ed_sktim': KNNED_SKTIME(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        #   'knn_dtw': KNNDTW(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        #   'knn_msm': KNNMSM(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'ee': EE(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        #   'tsf': TSF(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'ls': LS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        #   'st': ST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset']),
        #   'weasel': WEASEL(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        # 'cboss': CBOSS(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset']),
        'inception': INCEPTION(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim= args['dim'], ds= args['dataset'])
        # 'pforest': PFOREST(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset'])
    }

    logger.info(f"===== Step: Applying data splitting =====")
    split_indexes = get_split_indexes(X_train, args['split'])
    logger.info(
        f"Data split Indexes obtained. Time series length will be split into {args['split']} approximaltely equal chunks")

    dfs = []
    for i, index in enumerate(split_indexes):
        revealed_pct= (100 / args["split"])*(i+1)
        logger.info(f'######################################################')
        logger.info(
            f'===== Learning on {revealed_pct}% of the time series =====')
        X_train_splitted = apply_split(
            X_train, index, desc=arguments['--from_beg'])

        if args['cv']:
            analysis_data = cv(logger, models, X_train_splitted, X_test,
                               y_train, y_test)

        elif args['default_split']:
            analysis_data = no_cv(logger, models, X_train_splitted, X_test,
                                  y_train, y_test)

        logger.info(f"===== Step: Exporting fitted models =====")
        for m in models.values():
            export_model(m, 'etsc', revealed_pct)

        df = pd.DataFrame(analysis_data)
        df['revealed_pct'] = revealed_pct
        df['harmonic_mean'] = (2 * (1 - df['revealed_pct']) * df['test_ds_score']
                               )/((1 - df['revealed_pct']) + df['test_ds_score'])
        # df['rank_hm'] = df['harmonic_mean'].rank(
        #     ascending=False, method='dense')
        # df['rank_train_time'] = df['train_time'].rank(
        #     ascending=True, method='dense')
        # df['rank_score_time'] = df['test_ds_score_time'].rank(
        #     ascending=True, method='dense')
        dfs.append(df)
    analysis = pd.concat(dfs, axis=0, ignore_index=True)
    analysis['dataset'] = args['dataset']

    return analysis, models


def cv(logger, models, X_train, X_test, y_train, y_test):
    # # combine training and testing datasets
    # X = X_train.append(X_test, ignore_index=True)
    # y = np.concatenate((y_train, y_test), axis=0)

    logger.info(f"===== Step: Running Cross Validation =====")
    for m in models.values():
        m.fit(X_train, y_train)
        m.best_estimator_analysis['train_time'] = m.train_time

    logger.info(f"===== Step: Labeling Testing Dataset =====")
    for m in models.values():
        m.predict(X_test)

    logger.info(f"===== Step: Calculating Accuracy scores =====")
    for m in models.values():
        m.get_score(X_test, y_test)
        m.best_estimator_analysis['test_ds_score'] = m.test_ds_score
        m.best_estimator_analysis['test_ds_score_time'] = m.test_ds_score_time

    logger.info(f"===== Step: Prepare Analysis Results =====")
    analysis_data = [m.best_estimator_analysis for m in models.values()]
    logger.info(f"Analysis metrics: score, fitting time, testing time")
    # logger.info("models are ranked based on each of the metrics")
    return analysis_data


def no_cv(logger, models, X_train, X_test, y_train, y_test):
    logger.info(f"===== Step: Fitting models on Training Dataset =====")
    for m in models.values():
        m.fit(X_train, y_train)

    logger.info(f"===== Step: Labeling Testing Dataset =====")
    for m in models.values():
        m.predict(X_test)

    logger.info(f"===== Step: Calculating Accuracy scores =====")
    for m in models.values():
        m.get_score(X_test, y_test)

    logger.info(f"===== Step: Prepare Analysis Results =====")
    idx = ['classifier', 'train_time',
           'test_ds_score_time', 'test_ds_score', 'params']
    analysis_data = [pd.Series([m.clf_name, m.train_time, m.test_ds_score_time,
                                m.test_ds_score, m.clf.get_params()], index=idx) for m in models.values()]
    logger.info("Analysis metrics: score, fitting time, testing time")
    # logger.info("models are ranked based on each of the metrics")
    return analysis_data


def get_analysis_file_name(args):
    analysis_folder = os.path.join(os.getcwd(), 'analysis\\')
    process = 'tsc' if args['tsc'] else 'etsc'
    cv = 'cv' if args['cv'] else 'no_cv'
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    fname = os.path.join(analysis_folder, f"{args['dataset']}_{process}_{cv}")
    return fname


def get_model_analysis_file_name(process, dataset, cv, split, model_name):
    analysis_folder = os.path.join(
        os.getcwd(), 'datasets', dataset, 'analysis\\')
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


if __name__ == "__main__":
    analysis=None
    models=None
    logger=initialize_logger()
    try:
        arguments=docopt.docopt(__doc__)
        args=dict()
        args['tsc']=arguments['--tsc']
        args['etsc']=arguments['--etsc']
        args['cv']=arguments['--cv']
        args['default_split']=arguments['--default_split']
        args['dataset']=arguments['--dataset']
        args['n_iter']=int(arguments['--n_iter'])
        args['score_function']=arguments['--score_function']
        args['split']= None if arguments['--tsc'] else int(arguments['--split'])
        if arguments['--from_beg'] | (not arguments['--from_beg'] and not arguments['--from_end']):
            args['from_beg']=True
        args['from_end']=arguments['--from_end']

        logger.info(
            f"===== Step: Loading Dataset {arguments['--dataset']} =====")
        X_train, X_test, y_train, y_test=get_test_train_data(
            arguments['--dataset'])
        logger.info(f"Dataset {arguments['--dataset']} loaded !!")
        args['dim']=X_train.shape[1]
        if args['dim'] > 1:
            logger.info(f"Dataset {arguments['--dataset']} is Multivariate")
        else:
            logger.info(f"Dataset {arguments['--dataset']} is Univariate")

        if arguments['--tsc']:
            analysis, models=tsc(logger, args, X_train,
                                   X_test, y_train, y_test)
        elif arguments['--etsc']:
            analysis, models=etsc(
                logger, args, X_train, X_test, y_train, y_test)

        logger.info(f"===== Step: Exporting Analysis Results =====")
        export_analysis(analysis, args)
        # fname = get_analysis_file_name(args)
        # analysis.to_json(f'{fname}.json')
        # logger.info(f"Analysis exported to {fname}.json")

        import IPython
        IPython.embed()

    except Exception as e:
        logger.critical(e)
        raise
