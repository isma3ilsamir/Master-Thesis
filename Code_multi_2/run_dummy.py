"""
Usage:
    run.py --tsc (--dataset=<ds>)... (--cv [--n_iter=<n>] | --default_split) [--score_function=<f>]
    run.py --etsc (--dataset=<ds>)... (--cv [--n_iter=<n>] | --default_split) [--from_beg | --from_end] [--split=<s>] [--score_function=<f>]

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
    --split=<s>            The number of splits to apply to the time series. 10 splits= 10% increments, 20 splits= 5% increments,...etc [default: 10]
"""
import platform
import sys
import psutil
import time

from time import perf_counter
import docopt
import pandas as pd
import logging
import os
from datetime import datetime
import concurrent.futures
from itertools import repeat, count

from splitting import get_split_indexes, apply_split, get_split_indexes_w_threshold
from datasets import get_test_train_data, handle_missing_values

from sktime.classification.compose import ColumnEnsembleClassifier as ColEns

import matplotlib.pyplot as plt

from models import dummy

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

def initialize_models(args):
    logger.info(f"===== Step: Initializaing models =====")
    if args['tsc']:
        return {
            'dummy': dummy(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset'])
        }
    else:
        return {
            'dummy': dummy(scoring_function=args['score_function'], n_iter=args['n_iter'], cv=args['cv'], dim=args['dim'], ds=args['dataset'])
        }

def tsc(args, start_time, start= perf_counter()):
    try:
        logger.info(f"===== Starting Time Series Classification Model Selection Experiment =====")
        analysis_data = []
        ds= args['dataset']
        model= args['model']
        ts= datetime.strftime(start_time,"%d-%m-%Y %H:%M:%S")
        logger.info(f"===== Reading Data for {model.clf_name} =====")
        X_train, X_test, y_train, y_test= get_test_train_data(ds)
        handle_missing_values(X_train, ds, model.clf_name)
        handle_missing_values(X_test, ds, model.clf_name)

        if args['cv']:
            # This code is just handling for an error when doing cross-validation
            if isinstance(args['model'].clf, ColEns):
                ens_param= {'remainder': ['drop'], 'verbose': [False]}
                for k, v in ens_param.items():
                    for val in v:
                        args['model'].clf.set_params(**{k: val})
            analysis_data.append(cv('tsc', model, X_train, X_test, y_train, y_test, None))
        elif args['default_split']:
            analysis_data.append(no_cv('tsc', model, X_train, X_test, y_train, y_test, None))

        analysis = pd.DataFrame(analysis_data)
        analysis['dataset'] = args['dataset']

        logger.info(f"===== Step: Exporting Analysis Results for {model.clf_name} =====")
        export_analysis(analysis, args, start_time)
        stop= perf_counter()
        log= {
            'process': 'tsc',
            'ts': ts,
            'dataset': args['dataset'],
            'model': args['model'].clf_name,
            'cv': args['cv'],
            'n_iter': args['n_iter'],
            'score_function': args['score_function'],
            'from_beg': args['from_beg'],
            'split': args['split'],
            'revealed_pct': args['revealed_pct'],
            'success': True,
            'error': None,
            'duration': stop - start
        }

    except Exception as e:
        stop= perf_counter()
        ts= datetime.strftime(start_time,"%d-%m-%Y %H:%M:%S")
        logger.critical(f"===== tsc Failed for {args['model'].clf_name} =====")
        log={
            'process': 'tsc',
            'ts': ts,
            'dataset': args['dataset'],
            'model': args['model'].clf_name,
            'cv': args['cv'],
            'n_iter': args['n_iter'],
            'score_function': args['score_function'],
            'from_beg': args['from_beg'],
            'split': args['split'],
            'revealed_pct': args['revealed_pct'],
            'success': False,
            'error': None if not e else e,
            'duration': stop - start 
        }
    finally:
        logger.info(f"===== Step: Exporting Logs for {args['model'].clf_name} dataset {args['dataset']} =====")
        export_log(log, start_time)
        return log


def etsc(args, start_time, start= perf_counter()):
    try:
        start= perf_counter()
        logger.info(f"===== Starting Early Time Series Classification Experiment =====")
        analysis_data = []
        ds= args['dataset']
        model= args['model']
        revealed_pct= args['revealed_pct']
        ts= datetime.strftime(start_time,"%d-%m-%Y %H:%M:%S")
        logger.info(f"===== Reading Data for {model.clf_name} =====")
        X_train, X_test, y_train, y_test= get_test_train_data(ds)
        handle_missing_values(X_train, ds, model.clf_name)
        handle_missing_values(X_test, ds, model.clf_name)

        logger.info(f'===== Learning on {revealed_pct}% of the time series =====')
        X_train_splitted = apply_split(X_train, args['split_indexes'], asc=args['from_beg'])

        if args['cv']:
            # This code is just handling for an error when doing cross-validation
            if isinstance(args['model'].clf, ColEns):
                ens_param= {'remainder': ['drop'], 'verbose': [False]}
                for k, v in ens_param.items():
                    for val in v:
                        args['model'].clf.set_params(**{k: val})
            analysis_data.append(cv('etsc', model, X_train_splitted, X_test,
                            y_train, y_test, revealed_pct))

        elif args['default_split']:
            analysis_data.append(no_cv('etsc', model, X_train_splitted, X_test,
                                y_train, y_test, revealed_pct))

        analysis = pd.DataFrame(analysis_data)
        analysis['revealed_pct'] = revealed_pct
        analysis['harmonic_mean'] = (2 * (1 - analysis['revealed_pct']) * analysis['test_ds_score']) / ((1 - analysis['revealed_pct']) + analysis['test_ds_score'])
        analysis['dataset'] = args['dataset']
        logger.info(f"===== Step: Exporting Analysis Results for {model.clf_name} =====")
        export_analysis(analysis, args, start_time)

        stop= perf_counter()
        log= {
            'process': 'etsc',
            'ts': ts,
            'dataset': args['dataset'],
            'model': args['model'].clf_name,
            'cv': args['cv'],
            'n_iter': args['n_iter'],
            'score_function': args['score_function'],
            'from_beg': args['from_beg'],
            'split': args['split'],
            'revealed_pct': args['revealed_pct'],
            'success': True,
            'error': None,
            'duration': stop - start
        }

    except Exception as e:
        stop= perf_counter()
        ts= datetime.strftime(start_time,"%d-%m-%Y %H:%M:%S")
        logger.critical(f"=====etsc Failed for {args['model'].clf_name} dataset {args['dataset']} revealed percent {args['revealed_pct']}=====")
        log={
            'process': 'etsc',
            'ts': ts,
            'dataset': args['dataset'],
            'model': args['model'].clf_name,
            'cv': args['cv'],
            'n_iter': args['n_iter'],
            'score_function': args['score_function'],
            'from_beg': args['from_beg'],
            'split': args['split'],
            'revealed_pct': args['revealed_pct'],
            'success': False,
            'error': None if not e else e,
            'duration': stop - start
        }
    finally:
        logger.info(f"===== Step: Exporting Logs for {args['model'].clf_name} =====")
        export_log(log, start_time)
        return log

def cv(process, model, X_train, X_test, y_train, y_test, revealed_pct):
    logger.info(f"===== Step: Running Cross Validation for {model.clf_name} =====")
    model.fit(X_train, y_train)
    model.best_estimator_analysis['train_time'] = model.train_time

    # logger.info(f"===== Step: Labeling Testing Dataset for {model.clf_name} =====")
    # model.predict(X_test)

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


def no_cv(process, model, X_train, X_test, y_train, y_test, revealed_pct):
    logger.info(f"===== Step: Fitting models on Training Dataset for {model.clf_name} =====")
    model.fit(X_train, y_train)

    # logger.info(f"===== Step: Labeling Testing Dataset for {model.clf_name} =====")
    # model.predict(X_test)

    logger.info(f"===== Step: Calculating Accuracy scores for {model.clf_name} =====")
    model.get_score(X_test, y_test)

    logger.info(f"===== Step: Prepare Analysis Results for {model.clf_name} =====")
    idx = ['classifier', 'train_time','test_ds_score_time', 'test_ds_score', 'params']
    analysis= pd.Series([model.clf_name, model.train_time, model.test_ds_score_time, model.test_ds_score, model.clf.get_params()], index=idx)
    logger.info("Analysis metrics: score, fitting time, testing time")
    # # logger.info("models are ranked based on each of the metrics")

    logger.info(f"===== Step: Exporting fitted model for {model.clf_name} =====")
    export_model(model, process, revealed_pct)
    return analysis

def get_model_analysis_file_name(args, start_time):
    ts= datetime.strftime(start_time,"%Y_%m_%d_%H%M%S")
    process= 'tsc' if args['tsc'] else 'etsc'
    dataset= args['dataset']
    cv = 'cv' if args['cv'] else 'no_cv'
    pct = f"{int(args['revealed_pct'])}_pct" if args['revealed_pct'] else None
    model_name= args['model'].clf_name
    analysis_folder = os.path.join(
        os.getcwd(), 'datasets', dataset, 'analysis','')
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    f = f"{process}_{dataset}_{model_name}_{cv}_{pct}_{ts}" if pct else f"{process}_{dataset}_{model_name}_{cv}_{ts}"
    fname = os.path.join(analysis_folder, f)
    return fname

def export_analysis(analysis, args, start_time):
    model = args['model'].clf_name
    model_analysis_file=get_model_analysis_file_name(args, start_time)
    analysis.to_json(f'{model_analysis_file}.json')
    logger.info(f"Analysis for {model} exported to {model_analysis_file}.json")


def export_model(model, process, revealed_pct=None):
    try:
        if model.clf_name == 'PForest':
            logger.critical('Proximity Forest Model export is not currently working')
            return
        model.export_model(process,revealed_pct)
    except Exception as e:
        pct= 'full length'if not revealed_pct else f'{int(revealed_pct)}_pct'
        logger.critical(f'====== Exporting model {model.clf_name} for {pct} failed !!!! =======')

def get_log_file_name(start_time, logs):
    ts= datetime.strftime(start_time,"%Y_%m_%d_%H%M%S")
    process= logs['process']
    dataset= logs['dataset']
    cv = 'cv' if logs['cv'] else 'no_cv'
    pct = f"{int(logs['revealed_pct'])}_pct" if logs['revealed_pct'] else None
    model_name= logs['model']
    logs_folder = os.path.join(os.getcwd(), 'logs', '')
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    f = f"{process}_{dataset}_{model_name}_{cv}_{pct}_{ts}" if pct else f"{process}_{dataset}_{model_name}_{cv}_{ts}"
    fname = os.path.join(logs_folder, f)
    return fname

def export_log(logs, start_time):
    fname = get_log_file_name(start_time, logs)
    log_df = pd.DataFrame.from_dict([logs])
    log_df.to_json(f'{fname}.json')
    logger.info(f"Logs exported to {fname}.json")

def prepare_args(arguments, dataset):
    args={}
    args['tsc']=arguments['--tsc']
    args['etsc']=arguments['--etsc']
    args['dataset'] = dataset
    args['cv']=arguments['--cv']
    args['default_split']=arguments['--default_split']
    args['n_iter']= None if arguments['--default_split'] else int(arguments['--n_iter'])
    args['score_function']=arguments['--score_function']
    args['split']= None if arguments['--tsc'] else int(arguments['--split'])
    if arguments['--from_beg'] | (not arguments['--from_beg'] and not arguments['--from_end']):
        args['from_beg']=True
    else: args['from_beg'] = False
    args['from_end']=arguments['--from_end']
    X_train, _, _, _=get_test_train_data(args['dataset'])
    args['dim']=X_train.shape[1]
    args['models']= initialize_models(args)
    if not arguments['--tsc']:
        logger.info(f"===== Step: Applying data splitting for dataset {dataset}=====")
    args['split_indexes'] = None if arguments['--tsc'] else get_split_indexes(X_train, args['split'], dataset)
    ts_length= len(X_train['dim_0'][0])
    args['revealed_pct'] = None if arguments['--tsc'] else [int((100 * i)/ts_length) for i in args['split_indexes']]
    return args

def flatten_models(args):
    return list(map(prepare_model_args, repeat(args), args['models'].values()))

def prepare_model_args(args, model):
    new_args={}
    new_args['tsc']=args['tsc']
    new_args['etsc']=args['etsc']
    new_args['dataset'] = args['dataset']
    new_args['cv']=args['cv']
    new_args['default_split']=args['default_split']
    new_args['n_iter']= args['n_iter']
    new_args['score_function']= args['score_function']
    new_args['split']= args['split']
    new_args['from_beg']= args['from_beg']
    new_args['from_end']=args['from_end']
    new_args['dim']=args['dim']
    new_args['model']= model
    new_args['split_indexes']= args['split_indexes']
    new_args['revealed_pct']=   args['revealed_pct']
    return new_args

def flatten_split_indexes(args):
    return list(map(prepare_indexes_args, repeat(args), args['split_indexes'], args['revealed_pct']))

def prepare_indexes_args(args, index, revealed_pct):
    new_args={}
    new_args['tsc']=args['tsc']
    new_args['etsc']=args['etsc']
    new_args['dataset'] = args['dataset']
    new_args['cv']=args['cv']
    new_args['default_split']=args['default_split']
    new_args['n_iter']= args['n_iter']
    new_args['score_function']= args['score_function']
    new_args['split']= args['split']
    new_args['from_beg']= args['from_beg']
    new_args['from_end']=args['from_end']
    new_args['dim']=args['dim']
    new_args['model']= args['model']
    new_args['split_indexes']= index
    new_args['revealed_pct']= revealed_pct
    return new_args

def memory_limit(percentage: float):
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                ### commented code is the original, this is an enhancement
                # if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                #     free_memory += int(sline[1])
                free_memory = int(sline[1])
                break
    return free_memory

def memory(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator

# @memory(percentage=0.8)
def main():
    # max_workers= int(psutil.cpu_count() * 0.6)
    max_workers = 2
    start_time= datetime.now()
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

    # get flat args list
    args_list= list(map(prepare_args, repeat(arguments), arguments['--dataset']))
    args_list= list(map(flatten_models, args_list))
    flatten = lambda t: [item for sublist in t for item in sublist]
    flat_args_list= flatten(args_list)

    start = time.time()
    if arguments['--tsc']:
        with concurrent.futures.ProcessPoolExecutor(max_workers= max_workers) as executor:
            # futures = {executor.submit(tsc, flat_args_list, repeat(start_time))}
            futures = {
                executor.submit(tsc, task, start_time) : '_'.join([task['dataset'], task['model'].clf_name])
                    for task in flat_args_list
            }
            concurrent.futures.wait(futures)

    elif arguments['--etsc']:
        flat_split_list= list(map(flatten_split_indexes, flat_args_list))
        flatten = lambda t: [item for sublist in t for item in sublist]
        flat_split_list= flatten(flat_split_list)
        with concurrent.futures.ProcessPoolExecutor(max_workers= max_workers) as executor:
            # futures = {executor.submit(etsc, flat_split_list, repeat(start_time))}
            futures = {
                executor.submit(etsc, task, start_time) : '_'.join([task['dataset'], task['model'].clf_name, str(task['revealed_pct'])])
                    for task in flat_split_list
            }
            concurrent.futures.wait(futures)
    end = time.time()
 
    print("Total time : %ssecs" % (end - start))
 
    for fut in concurrent.futures.as_completed(futures):
        print(f"The outcome for {fut} is {fut.result()}")

if __name__ == "__main__":
    main()