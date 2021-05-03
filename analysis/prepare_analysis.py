import csv
import itertools
import math
import Orange
from cd_rank import wilcoxon_holm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import IPython
from pandas.io import json
from csv import reader
import pathlib
import glob
import pandas as pd
import sys
import os
from numpy.lib.function_base import median
import random
curr_folder = os.path.dirname(os.path.realpath(__file__))
if curr_folder not in sys.path:
    sys.path.insert(0, curr_folder)

MAIN_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def round_down(num, divisor):
    return num - (num % divisor)


def get_error(df):
    df_copy = df.copy()
    df_copy['error_rate'] = 1 - df_copy['test_ds_score']
    return df_copy


def get_rev_hm(row, beta=1):
    revealed_pct = row['revealed_pct_actual']
    err = row['error_rate']
    rev_hm = None
    if err == 1 and revealed_pct == 0:
        rev_hm = 1
    elif err == 1 and revealed_pct == 1:
        rev_hm = 1
    else:
        rev_hm = (1 + pow(beta, 2)) * ((err * (revealed_pct/100)) /
                                       ((pow(beta, 2) * err) + (revealed_pct/100)))
        # rev_hm = (2 * (revealed_pct/100) * err) / ((revealed_pct/100) + err)
    return rev_hm


def adjust_hm(df):
    df_copy = df.copy()
    df_copy['harmonic_mean'] = 1 - df_copy['reversed_hm']
    return df_copy

# def adjust_hm(df):
#     df_copy = df.copy()
#     df_copy['harmonic_mean'] = (2 * (1 - (df_copy['revealed_pct_actual']/100)) * df_copy['test_ds_score']) / ((1 - (df_copy['revealed_pct_actual']/100)) + df_copy['test_ds_score'])
#     return df_copy


def create_whole_revpct(df):
    df['revealed_pct_actual'] = df['revealed_pct']
    # round revealed_pct
    df['revealed_pct'] = round_down(df['revealed_pct_actual'], 10)
    return df


def create_clf_pct_col(df):
    df = create_whole_revpct(df)
    # id dummy clf is dummy else use naming convention
    df['clf'] = df['classifier'] + '_' + df['revealed_pct'].astype(str)
    return df


def get_joined_df(df, dataset_df):
    joined = df.set_index('dataset').join(
        dataset_df.set_index('dataset'), how='inner')
    joined = joined.reset_index(drop=False)
    return joined


def filter_by_val(df, col, val):
    return df[df[col] == val]


def filter_by_list(df, col, values):
    return df[df[col].isin(values)]

def filter_by_not_in_list(df, col, values):
    return df[~df[col].isin(values)]

def get_datasets_df(ds_file='Datasets_metadata.csv', index_col=0, sheet='Sheet3'):
    # return pd.read_excel(ds_file, index_col= index_col, sheet_name= sheet)
    ds_metadata_path = os.path.join(MAIN_DIRECTORY, ds_file)
    return pd.read_csv(ds_metadata_path)


def get_analysis_df(df):
    cols = ['classifier', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
            'std_score_time', 'params', 'mean_test_score', 'std_test_score',
            'train_time', 'test_ds_score', 'test_ds_score_time',
            'revealed_pct', 'harmonic_mean', 'dataset']
    return df[cols].copy()


def remove_duplicates(df):
    cols = df.columns.drop('params')
    df_1 = df[cols].copy()
    df_1.drop_duplicates(inplace=True)
    unique_indexes = df_1.index
    df = df.filter(items=unique_indexes, axis='index')
    df.reset_index(inplace=True)
    return df


def get_json_files(analysis_filenames):
    json_files = []
    analysis_names_path = os.path.join(curr_folder, analysis_filenames)
    with open(analysis_names_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[0] == 'filename':
                continue
            file_name = os.path.basename(row[0])
            # analysis_path = glob.glob(os.path.join(os.getcwd(),f'*/datasets/*/analysis/{file_name}'))
            analysis_path = glob.glob(
                os.path.join(MAIN_DIRECTORY, '*', file_name))
            if not analysis_path:
                print(f'there was no analysis file found for {row[0]}')
                continue
                # raise Exception(f'there was no analysis file found for {row[0]}')
            elif len(analysis_path) > 1:
                print(f'there is more than one location for file {row[0]}:')
                print(analysis_path)
            json_files.append(analysis_path[0])
    return json_files


def extract_json_data(json_files):
    dfs = []
    for j in json_files:
        df = pd.read_json(j)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)


def get_cd_df_acc(df):
    # for CD diagram
    cd_df = df[['clf', 'dataset', 'test_ds_score']]
    cd_df.columns = ['classifier_name', 'dataset_name', 'accuracy']
    return cd_df


def get_cd_df_hm(df):
    # for CD diagram
    cd_df = df[['clf', 'dataset', 'harmonic_mean']]
    cd_df.columns = ['classifier_name', 'dataset_name', 'accuracy']
    return cd_df


def get_cd_diagram(df, ds_list, metric, image_name):
    cd_df = None
    ds_len = len(ds_list)
    if metric == 'accuracy':
        cd_df = get_cd_df_acc(df)
    elif metric == 'hm':
        cd_df = get_cd_df_hm(df)
    else:
        raise("metric should be either accuracy or hm")
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=cd_df)

    names = average_ranks.index.tolist()
    avranks = average_ranks.tolist()
    # calculate CD
    # cd = Orange.evaluation.compute_CD(avranks, ds_len)
    q005 = 3.7145
    N = len(ds_list)
    k = len(names)
    cd = q005*(math.sqrt((k*(k+1))/(6*N)))
    Orange.evaluation.graph_ranks(
        avranks, names, cd=cd, width=6, textspace=1.5, reverse=True)
    # plt.show()
    export_folder = os.path.join(curr_folder, 'cd_diagrams')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(export_folder, image_name), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()
    average_ranks.to_csv(os.path.join(
        export_folder, f'{image_name}_avg_ranks.csv'))


def rank_on_ds_score(df):
    dfs = []
    datasets = list(df['dataset'].unique())
    for ds in datasets:
        filtered = filter_by_val(df, 'dataset', ds).copy()
        filtered['ds_rank_accuracy'] = filtered['test_ds_score'].rank(
            ascending=False, method='min')
        dfs.append(filtered)
    ranked = pd.concat(dfs, axis=0, ignore_index=True)
    return ranked


def rank_on_hm(df):
    dfs = []
    datasets = list(df['dataset'].unique())
    for ds in datasets:
        filtered = filter_by_val(df, 'dataset', ds).copy()
        filtered['ds_rank_hm'] = filtered['harmonic_mean'].rank(
            ascending=False, method='min')
        dfs.append(filtered)
    ranked = pd.concat(dfs, axis=0, ignore_index=True)
    return ranked


def get_used_train_length(row, splits=10):
    length = row['length']
    pct = row['revealed_pct']
    split_lengths = ([length // splits + (1 if x < length %
                                          splits else 0) for x in range(splits)])
    split_indexes = np.cumsum(split_lengths).tolist()
    return split_indexes[int(pct/10) - 1]


def get_standard_train_length(row):
    # pct10_length = [5,10,25,50,100]
    # pct20_length = [10,20,50,100,200]
    # pct30_length = [15,30,75,150,300]
    pct100_length = [50, 100, 250, 500, 1000]
    pct = row['revealed_pct']
    if pct == 10:
        train_length_std = apply_metric_standardization(
            row, pct100_length, 'length')
    elif pct == 20:
        train_length_std = apply_metric_standardization(
            row, pct100_length, 'length')
    elif pct == 30:
        train_length_std = apply_metric_standardization(
            row, pct100_length, 'length')
    elif pct == 100:
        train_length_std = apply_metric_standardization(
            row, pct100_length, 'length')
    else:
        raise Exception(f"No handling for {pct}% in length std")
    return train_length_std


def get_standard_train_size(row):
    sizes = [50, 100, 250, 500, 1000]
    train_size_std = apply_metric_standardization(row, sizes, 'train_size')
    return train_size_std


def get_standard_num_classes(row):
    classes = [2, 3, 5, 10]
    if row['num_classes'] in [2, 3]:
        train_size_std = row['num_classes']
    else:
        train_size_std = apply_metric_standardization(
            row, classes, 'num_classes')
    return train_size_std


def apply_metric_standardization(row, values, col):
    r = len(values) + 1
    for i in range(r):
        if i == 0:
            min = 0
            max = values[i]
        elif i == len(values):
            min = values[i-1]
            max = np.inf
        else:
            min = values[i-1]
            max = values[i]

        if (row[col] > min) and (row[col] <= max):
            if max != np.inf:
                return f"{min+1}-{max}"
            else:
                return f'{min+1} +'
        else:
            continue


def get_ds_finished_chunks_for_clf(df, classifier, num_chunks):
    df = filter_by_val(df, 'classifier', classifier)
    ds_chunks = df.groupby('dataset')[
        'revealed_pct'].size().to_frame().reset_index()
    ds_chunks.columns = ['dataset', 'chunks_finished']
    ds_list = filter_by_val(ds_chunks, 'chunks_finished', num_chunks)[
        'dataset'].tolist()
    return ds_list


def get_ds_finished_clfs_for_revpct(df, revealed_pct, num_classifiers):
    df = filter_by_val(df, 'revealed_pct', revealed_pct)
    ds_clfs = df.groupby('dataset')[
        'classifier'].size().to_frame().reset_index()
    ds_clfs.columns = ['dataset', 'classifiers_finished']
    ds_list = filter_by_val(ds_clfs, 'classifiers_finished', num_classifiers)[
        'dataset'].tolist()
    return ds_list


def get_extended_datasets(dataset_df, revealed_pct, classifiers):
    # get all datasets with revealed_pct and runs output
    datasets_extended = pd.concat([dataset_df['dataset']] * len(revealed_pct), keys=revealed_pct).reset_index(
        level=1, drop=True).rename_axis('revealed_pct').reset_index()
    datasets_extended = pd.concat([datasets_extended] * len(classifiers),
                                  keys=classifiers).reset_index(level=1, drop=True).rename_axis('model').reset_index()
    return datasets_extended


def get_extended_dataset_df(dataset_df, pcts, classifiers):
    dataset_extended = get_extended_datasets(dataset_df, pcts, classifiers)
    return dataset_extended


def get_dummy_for_all_pcts(df):
    df_dummy = df[df['classifier'] == 'Dummy'].copy()
    pcts = [10, 20, 30]
    dfs = []
    for pct in pcts:
        dummy_pct = df_dummy.copy()
        dummy_pct['revealed_pct'] = pct
        dfs.append(dummy_pct)
    dummy_data = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    df_final = pd.concat([df, dummy_data], axis=0,
                         ignore_index=True, sort=False)
    return df_final


def get_main_df(dataset_df, analysis_filenames):
    json_files = get_json_files(analysis_filenames)
    df = extract_json_data(json_files)
    df = get_analysis_df(df)
    df = get_dummy_for_all_pcts(df)
    df = create_clf_pct_col(df)
    df = get_joined_df(df, dataset_df)
    df['actual_train_length'] = df.apply(get_used_train_length, axis=1)
    df['train_length_std'] = df.apply(get_standard_train_length, axis=1)
    df['train_size_std'] = df.apply(get_standard_train_size, axis=1)
    df['num_classes_std'] = df.apply(get_standard_num_classes, axis=1)
    df = get_error(df)
    df['reversed_hm'] = df.apply(get_rev_hm, beta=1, axis=1)
    df = adjust_hm(df)
    return df


def get_pivot(df, val_col, cols, idx, agg):
    if agg == 'sum':
        aggfunc = np.sum
    elif agg == 'mean':
        aggfunc = np.mean
    elif agg == 'median':
        aggfunc = np.median
    else:
        raise('agg should be sum or mean')
    return pd.pivot_table(df, values=val_col, columns=cols, aggfunc=aggfunc, index=idx)


def get_df_finished_all_by_pct(df, rev_pct, num_classifiers=5):
    ds_list = get_ds_finished_clfs_for_revpct(df, rev_pct, num_classifiers)
    df_pct_data = filter_by_val(df, 'revealed_pct', rev_pct)
    df_pct_filtered = filter_by_list(df_pct_data, 'dataset', ds_list)
    return df_pct_filtered, ds_list


def get_df_finished_all_pct_by_clf(df, classifier, num_chunks=4):
    ds_list = get_ds_finished_chunks_for_clf(df, classifier, num_chunks)
    df_clf_data = filter_by_val(df, 'classifier', classifier)
    df_clf_filtered = filter_by_list(df_clf_data, 'dataset', ds_list)
    return df_clf_filtered, ds_list


def get_df_finished_all_pct_all_clf(df):
    ds_10pct_list = get_ds_finished_clfs_for_revpct(df, 10, 5)
    ds_20pct_list = get_ds_finished_clfs_for_revpct(df, 20, 5)
    ds_30pct_list = get_ds_finished_clfs_for_revpct(df, 30, 5)
    ds_100pct_list = get_ds_finished_clfs_for_revpct(df, 100, 5)
    ds_all_pct_all_clf = set(ds_10pct_list).intersection(
        set(ds_20pct_list), set(ds_30pct_list), set(ds_100pct_list))
    df_all = filter_by_list(df, 'dataset', ds_all_pct_all_clf)
    return df_all, ds_all_pct_all_clf


def get_scatter_plot_same_clf_two_pct(df, pct1, pct2, clf_name, metric, image_name):
    # scatter plot for comparing 2 percentages from same classifier
    # df_pivot= pd.pivot_table(df, values='test_ds_score', columns=['revealed_pct'], aggfunc=np.sum, index=['dataset'])
    df_pivot = get_pivot(df, metric, 'revealed_pct', 'dataset', 'sum')
    plt.plot(df_pivot[pct1], df_pivot[pct2], 'o', color='black')
    # plt.plot([0,1], [0, 1], 'k-')
    plt.xlabel(f'{clf_name}_{pct1}', fontsize=16)
    # plt.margins(x=0)
    plt.ylabel(f'{clf_name}_{pct2}', fontsize=16)
    # plt.margins(y=0)
    plt.fill([0, 0, 1, 0], [0, 1, 1, 0], 'lightskyblue',
             alpha=0.2, edgecolor='lightskyblue')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plt.show()
    export_folder = os.path.join(curr_folder, 'scatter_plots')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(export_folder, image_name), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


def get_boxplot_all_pct_by_clf(df, metric, image_name):
    if metric == 'accuracy':
        # sns.catplot(kind='box', y="test_ds_score", x="clf", data=df, showfliers = False)
        box_plot = sns.catplot(kind='box', y="test_ds_score", x="clf", data=df.sort_values(
            "revealed_pct"), flierprops=dict(markerfacecolor='0.50', markersize=2))
        medians = df.groupby(['revealed_pct'])[
            'test_ds_score'].median().round(3)
        # offset from median for display
        vertical_offset = df['test_ds_score'].median() * 0.03
    elif metric == 'hm':
        # sns.catplot(kind='box', y="harmonic_mean", x="clf", data=df, showfliers = False)
        box_plot = sns.catplot(kind='box', y="harmonic_mean", x="clf", data=df.sort_values(
            "revealed_pct"), flierprops=dict(markerfacecolor='0.50', markersize=2))
        medians = df.groupby(['revealed_pct'])[
            'harmonic_mean'].median().round(3)
        # offset from median for display
        vertical_offset = df['harmonic_mean'].median() * 0.03
    elif metric == 'train_time':
        # sns.catplot(kind='box', y="train_time", x="clf", data=df, showfliers = False)
        box_plot = sns.catplot(kind='box', y="train_time", x="clf", data=df.sort_values(
            "revealed_pct"), flierprops=dict(markerfacecolor='0.50', markersize=2))
        medians = df.groupby(['revealed_pct'])['train_time'].median().round(3)
        # offset from median for display
        vertical_offset = df['train_time'].median() * 0.03
    else:
        raise("metric should be accuracy, train_time or hm")
    for xtick in box_plot.ax.get_xticks():
        box_plot.ax.text(xtick, medians.iloc[xtick] + vertical_offset, medians.iloc[xtick],
                         horizontalalignment='center', size='x-small', color='w', weight='semibold')
    export_folder = os.path.join(curr_folder, 'box_plots')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(export_folder, image_name), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


def export_results_df(df, metric, file_name):
    if metric == 'test_ds_score':
        df_result = get_pivot(df, 'test_ds_score', 'clf', 'dataset', 'mean')
    elif metric == 'harmonic_mean':
        df_result = get_pivot(df, 'harmonic_mean', 'clf', 'dataset', 'mean')
    export_folder = os.path.join(curr_folder, 'results_tables')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    df_result.to_csv(os.path.join(export_folder, file_name))


def get_count_first_rank_by_feature(df, feature, measure, pcts):
    for p in pcts:
        mask = df['revealed_pct'] == p
        df_temp = df[mask].copy()
        if measure == 'accuracy':
            df_temp_ranked = rank_on_ds_score(df_temp)
        elif measure == 'hm':
            df_temp_ranked = rank_on_hm(df_temp)
        df_temp_ranked[f'ds_rank_{measure}_calc'] = np.where(
            df_temp_ranked[f'ds_rank_{measure}'] == 1, 1, 0)
        values_list = df_temp_ranked[feature].unique().tolist()
        dfs = []
        for val in values_list:
            num_ds = len(
                df_temp_ranked[df_temp_ranked[feature] == val]['dataset'].unique())
            res = df_temp_ranked[df_temp_ranked[feature] == val].groupby(
                'classifier', as_index=False)[[f'ds_rank_{measure}_calc']].sum()
            res_pvt = get_pivot(
                res, f'ds_rank_{measure}_calc', 'classifier', None, 'sum')
            res_pvt.index = [val]
            res_pvt['num_datasets'] = num_ds
            dfs.append(res_pvt)
        export_folder = os.path.join(curr_folder, 'results_tables')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        df_avg_rank = pd.concat(dfs, axis=0, sort=True)
        df_avg_rank.reset_index(inplace=True)
        df_avg_rank.rename({'index': feature}, axis='columns', inplace=True)
        df_avg_rank.columns.name = None
        df_avg_rank = df_avg_rank.round(2)
        df_avg_rank.to_csv(os.path.join(
            export_folder, f'count_rank_first_{p}pct_by_{feature}_{measure}.csv'), index=False)


def get_count_first_rank_by_feature_with_dummy(df, feature, measure, pcts, df_dummy):
    for p in pcts:
        mask = df['revealed_pct'] == p
        df_temp = df[mask].copy()
        ds_list = df_temp['dataset'].unique().tolist()
        df_dummy_pct = filter_by_list(df_dummy, 'dataset', ds_list).copy()
        df_dummy_pct = filter_by_val(df_dummy_pct, 'revealed_pct', p)
        data_with_dummy = pd.concat(
            [df_temp, df_dummy_pct], axis=0, ignore_index=True, sort=False)
        if measure == 'accuracy':
            df_temp_ranked = rank_on_ds_score(data_with_dummy)
        elif measure == 'hm':
            df_temp_ranked = rank_on_hm(data_with_dummy)
        df_temp_ranked = rank_relative_to_dummy(df_temp_ranked, measure)
        values_list = df_temp_ranked[feature].unique().tolist()
        dfs = []
        for val in values_list:
            num_ds = len(
                df_temp_ranked[df_temp_ranked[feature] == val]['dataset'].unique())
            res = df_temp_ranked[df_temp_ranked[feature] == val].groupby(
                'classifier', as_index=False)[[f'ds_rank_{measure}_calc']].sum()
            res_pvt = get_pivot(
                res, f'ds_rank_{measure}_calc', 'classifier', None, 'sum')
            res_pvt.index = [val]
            res_pvt['num_datasets'] = num_ds
            dfs.append(res_pvt)
        export_folder = os.path.join(curr_folder, 'results_tables')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        df_avg_rank = pd.concat(dfs, axis=0, sort=True)
        df_avg_rank.reset_index(inplace=True)
        df_avg_rank.rename({'index': feature}, axis='columns', inplace=True)
        df_avg_rank.columns.name = None
        df_avg_rank = df_avg_rank.round(2)
        df_avg_rank.to_csv(os.path.join(
            export_folder, f'count_rank_first_{p}pct_by_{feature}_{measure}_with_dummy.csv'), index=False)


def rank_relative_to_dummy(df, measure):
    dfs = []
    datasets = list(df['dataset'].unique())
    for ds in datasets:
        filtered = filter_by_val(df, 'dataset', ds).copy()
        dummy_rank = filtered[filtered['classifier']
                              == 'Dummy'][f'ds_rank_{measure}'].iloc[0]
        filtered[f'ds_rank_{measure}_calc'] = np.where(
            filtered[f'ds_rank_{measure}'] >= dummy_rank, 0, 1)
        dfs.append(filtered)
    ranked = pd.concat(dfs, axis=0, ignore_index=True)
    return ranked


def get_avg_rank_by_feature(df, feature, measure, pcts):
    for p in pcts:
        mask = df['revealed_pct'] == p
        df_temp = df[mask].copy()
        if measure == 'accuracy':
            df_temp_ranked = rank_on_ds_score(df_temp)
        elif measure == 'hm':
            df_temp_ranked = rank_on_hm(df_temp)
        values_list = df_temp_ranked[feature].unique().tolist()
        dfs = []
        for val in values_list:
            num_ds = len(
                df_temp_ranked[df_temp_ranked[feature] == val]['dataset'].unique())
            res = df_temp_ranked[df_temp_ranked[feature] == val].groupby(
                'classifier', as_index=False)[[f'ds_rank_{measure}']].mean()
            res_pvt = get_pivot(
                res, f'ds_rank_{measure}', 'classifier', None, 'sum')
            res_pvt.index = [val]
            res_pvt['num_datasets'] = num_ds
            dfs.append(res_pvt)
        export_folder = os.path.join(curr_folder, 'results_tables')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        df_avg_rank = pd.concat(dfs, axis=0, sort=True)
        df_avg_rank.reset_index(inplace=True)
        df_avg_rank.rename({'index': feature}, axis='columns', inplace=True)
        df_avg_rank.columns.name = None
        df_avg_rank = df_avg_rank.round(2)
        df_avg_rank.to_csv(os.path.join(
            export_folder, f'avg_rank_{p}pct_by_{feature}_{measure}.csv'), index=False)


def get_avg_rank_by_feature_with_dummy(df, feature, measure, pcts, df_dummy):
    for p in pcts:
        mask = df['revealed_pct'] == p
        df_temp = df[mask].copy()
        ds_list = df_temp['dataset'].unique().tolist()
        df_dummy_pct = filter_by_list(df_dummy, 'dataset', ds_list).copy()
        df_dummy_pct = filter_by_val(df_dummy_pct, 'revealed_pct', p)
        data_with_dummy = pd.concat(
            [df_temp, df_dummy_pct], axis=0, ignore_index=True, sort=False)
        if measure == 'accuracy':
            df_temp_ranked = rank_on_ds_score(data_with_dummy)
        elif measure == 'hm':
            df_temp_ranked = rank_on_hm(data_with_dummy)
        values_list = df_temp_ranked[feature].unique().tolist()
        dfs = []
        for val in values_list:
            num_ds = len(
                df_temp_ranked[df_temp_ranked[feature] == val]['dataset'].unique())
            res = df_temp_ranked[df_temp_ranked[feature] == val].groupby(
                'classifier', as_index=False)[[f'ds_rank_{measure}']].mean()
            res_pvt = get_pivot(
                res, f'ds_rank_{measure}', 'classifier', None, 'sum')
            res_pvt.index = [val]
            res_pvt['num_datasets'] = num_ds
            dfs.append(res_pvt)
        export_folder = os.path.join(curr_folder, 'results_tables')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        df_avg_rank = pd.concat(dfs, axis=0, sort=True)
        df_avg_rank.reset_index(inplace=True)
        df_avg_rank.rename({'index': feature}, axis='columns', inplace=True)
        df_avg_rank.columns.name = None
        df_avg_rank = df_avg_rank.round(2)
        df_avg_rank.to_csv(os.path.join(
            export_folder, f'avg_rank_{p}pct_by_{feature}_{measure}_with_dummy.csv'), index=False)

def export_data_for_recommendation(df):
    df_export = df[['dataset',
                    'classifier',
                    # 'test_ds_score',
                    'revealed_pct',
                    'train_size',
                    'test_size',
                    'num_classes',
                    'type',
                    'num_dim',
                    'balanced',
                    'length',
                    'harmonic_mean'
                    ]].copy()
    export_folder = os.path.join(os.path.dirname(curr_folder), 'recommendation')
    df_export.to_csv(os.path.join(
            export_folder,'recommendation_data.csv'), index= False)

if __name__ == "__main__":
    # fundamental values
    pcts = [10, 20, 30, 100]
    classifiers = ['ST', 'CBoss', 'TSF', 'PForest', 'WEASEL', 'Dummy']
    analysis_filenames = 'log_filenames.csv'

    # get datasets info
    dataset_df = get_datasets_df()
    dataset_extended = get_extended_dataset_df(dataset_df, pcts, classifiers)

    # get analysis results for all
    df_main = get_main_df(dataset_df, analysis_filenames)

    # separate dummy results
    df_dummy = filter_by_val(df_main, 'classifier', 'Dummy')

    # df for other classifiers
    df = df_main[df_main['classifier'] != 'Dummy']
    # df filtered by percent
    df_10 = filter_by_val(df, 'revealed_pct', 10)
    df_20 = filter_by_val(df, 'revealed_pct', 20)
    df_30 = filter_by_val(df, 'revealed_pct', 30)
    df_100 = filter_by_val(df, 'revealed_pct', 100)
    # df filtered by classifier
    clf_df_list = []
    df_tsf = filter_by_val(df, 'classifier', 'TSF')
    clf_df_list.append((df_tsf, 'TSF'))
    df_st = filter_by_val(df, 'classifier', 'ST')
    clf_df_list.append((df_st, 'ST'))
    df_cbs = filter_by_val(df, 'classifier', 'CBoss')
    clf_df_list.append((df_cbs, 'CBoss'))
    df_wsl = filter_by_val(df, 'classifier', 'WEASEL')
    clf_df_list.append((df_wsl, 'WEASEL'))
    df_pfrst = filter_by_val(df, 'classifier', 'PForest')
    clf_df_list.append((df_pfrst, 'PForest'))

    ###### Finished by percent all classifiers ######
    same_pct_all_clf_list = []
    ## 10% datasets
    df_all_10pct_data, df_all_10pct_ds_list = get_df_finished_all_by_pct(
        df, 10, 5)
    same_pct_all_clf_list.append(
        (df_all_10pct_data, df_all_10pct_ds_list, '10pct'))
    ## 20% datasets
    df_all_20pct_data, df_all_20pct_ds_list = get_df_finished_all_by_pct(
        df, 20, 5)
    same_pct_all_clf_list.append(
        (df_all_20pct_data, df_all_20pct_ds_list, '20pct'))
    ## 30% datasets
    df_all_30pct_data, df_all_30pct_ds_list = get_df_finished_all_by_pct(
        df, 30, 5)
    same_pct_all_clf_list.append(
        (df_all_30pct_data, df_all_30pct_ds_list, '30pct'))
    ## 100% datasets
    df_all_100pct_data, df_all_100pct_ds_list = get_df_finished_all_by_pct(
        df, 100, 5)
    same_pct_all_clf_list.append(
        (df_all_100pct_data, df_all_100pct_ds_list, '100pct'))
    # all percentages
    df_all_pct_all_clf, df_all_pct_all_clf_ds_list = get_df_finished_all_pct_all_clf(
        df)

    ###### Finished by classifier all pct ######
    all_pct_same_clf_list = []
    # TSF
    df_all_pct_tsf_data, df_all_pct_tsf_ds_list = get_df_finished_all_pct_by_clf(
        df, 'TSF', 4)
    all_pct_same_clf_list.append(
        (df_all_pct_tsf_data, df_all_pct_tsf_ds_list, 'tsf'))
    # ST
    df_all_pct_st_data, df_all_pct_st_ds_list = get_df_finished_all_pct_by_clf(
        df, 'ST', 4)
    all_pct_same_clf_list.append(
        (df_all_pct_st_data, df_all_pct_st_ds_list, 'st'))
    # CBoss
    df_all_pct_cbs_data, df_all_pct_cbs_ds_list = get_df_finished_all_pct_by_clf(
        df, 'CBoss', 4)
    all_pct_same_clf_list.append(
        (df_all_pct_cbs_data, df_all_pct_cbs_ds_list, 'cboss'))
    # WEASEL
    df_all_pct_wsl_data, df_all_pct_wsl_ds_list = get_df_finished_all_pct_by_clf(
        df, 'WEASEL', 4)
    all_pct_same_clf_list.append(
        (df_all_pct_wsl_data, df_all_pct_wsl_ds_list, 'weasel'))
    # PForest
    df_all_pct_pfrst_data, df_all_pct_pfrst_ds_list = get_df_finished_all_pct_by_clf(
        df, 'PForest', 4)
    all_pct_same_clf_list.append(
        (df_all_pct_pfrst_data, df_all_pct_pfrst_ds_list, 'pforest'))

    ###### table of results ######
    export_results_df(df, 'test_ds_score', 'accuracies_results.csv')
    export_results_df(df, 'harmonic_mean', 'hm_results.csv')

    # for p in pcts:
    #     mask = df['revealed_pct'] == p
    #     df_temp = df[mask].copy()
    #     IPython.embed()
    #     df_temp_ranked = rank_on_ds_score(df_temp)
    #     df_temp_ranked = rank_on_hm(df_temp)

    # ######## AVG Ranking #############

    # ###### Ranking each percent by dimension (tables exported as csv) ######
    # ### by type
    # # get_avg_rank_by_feature(df_all_pct_all_clf, 'type', 'accuracy', pcts)
    # get_avg_rank_by_feature(df_all_pct_all_clf, 'type', 'hm', pcts)
    # ### by length
    # # get_avg_rank_by_feature(df_all_pct_all_clf, 'train_length_std', 'accuracy', pcts)
    # get_avg_rank_by_feature(df_all_pct_all_clf, 'train_length_std', 'hm', pcts)
    # ### by train size
    # # get_avg_rank_by_feature(df_all_pct_all_clf, 'train_size_std', 'accuracy', pcts)
    # get_avg_rank_by_feature(df_all_pct_all_clf, 'train_size_std', 'hm', pcts)
    # ### by num classes
    # # get_avg_rank_by_feature(df_all_pct_all_clf, 'num_classes_std', 'accuracy', pcts)
    # get_avg_rank_by_feature(df_all_pct_all_clf, 'num_classes_std', 'hm', pcts)

    # ###### Ranking each percent by dimension with Dummy (tables exported as csv) ######
    # ### by type
    # get_avg_rank_by_feature_with_dummy(df_all_pct_all_clf, 'type', 'hm', pcts, df_dummy)
    # ### by length
    # get_avg_rank_by_feature_with_dummy(df_all_pct_all_clf, 'train_length_std', 'hm', pcts, df_dummy)
    # ### by train size
    # get_avg_rank_by_feature_with_dummy(df_all_pct_all_clf, 'train_size_std', 'hm', pcts, df_dummy)
    # ### by num classes
    # get_avg_rank_by_feature_with_dummy(df_all_pct_all_clf, 'num_classes_std', 'hm', pcts, df_dummy)

    ###### Ranking each percent by dimension with Dummy (tables exported as csv) ######
    # by type
    get_count_first_rank_by_feature_with_dummy(
        df_all_pct_all_clf, 'type', 'hm', pcts, df_dummy)
    # by length
    get_count_first_rank_by_feature_with_dummy(
        df_all_pct_all_clf, 'train_length_std', 'hm', pcts, df_dummy)
    # by train size
    get_count_first_rank_by_feature_with_dummy(
        df_all_pct_all_clf, 'train_size_std', 'hm', pcts, df_dummy)
    # by num classes
    get_count_first_rank_by_feature_with_dummy(
        df_all_pct_all_clf, 'num_classes_std', 'hm', pcts, df_dummy)

    ###### boxplot distribution ######
    for data, clf in clf_df_list:
        get_boxplot_all_pct_by_clf(data, 'accuracy', f'boxplot_accuracy_{clf}')
        get_boxplot_all_pct_by_clf(data, 'hm', f'boxplot_hm_{clf}')
        get_boxplot_all_pct_by_clf(
            data, 'train_time', f'boxplot_train_time_{clf}')

    ###### CD ######
    # within classifiers
    for data, ds_list, clf in all_pct_same_clf_list:
        get_cd_diagram(data, ds_list, 'accuracy', f'cd_accuracy_within_{clf}')
        get_cd_diagram(data, ds_list, 'hm', f'cd_hm_within_{clf}')

    # between classifiers (same percent WITHOUT dummy)
    for data, ds_list, pct in same_pct_all_clf_list:
        df_dummy_pct = filter_by_list(df_dummy, 'dataset', ds_list).copy()
        df_dummy_pct = filter_by_val(
            df_dummy_pct, 'revealed_pct', int(pct[:-3]))
        data_with_dummy = pd.concat(
            [data, df_dummy_pct], axis=0, ignore_index=True, sort=False)
        get_cd_diagram(data_with_dummy, ds_list, 'accuracy',
                       f'cd_accuracy_across_{pct}_with_dummy')
        get_cd_diagram(data_with_dummy, ds_list, 'hm',
                       f'cd_hm_across_{pct}_with_dummy')

    # between classifiers (same percent with dummy)
    for data, ds_list, pct in same_pct_all_clf_list:
        get_cd_diagram(data, ds_list, 'accuracy', f'cd_accuracy_across_{pct}')
        get_cd_diagram(data, ds_list, 'hm', f'cd_hm_across_{pct}')

    # between classifiers (all WITHOUT dummy)
    get_cd_diagram(df_all_pct_all_clf, df_all_pct_all_clf_ds_list,
                   'accuracy', f'cd_accuracy_all_pct_all_clf')
    get_cd_diagram(df_all_pct_all_clf, df_all_pct_all_clf_ds_list,
                   'hm', f'cd_hm_all_pct_all_clf')

    # between classifiers (all with dummy)
    df_dummy_join_with_all = filter_by_list(
        df_dummy, 'dataset', df_all_pct_all_clf_ds_list)
    # df_dummy_join_with_all[df_dummy_join_with_all['revealed_pct']==10]
    df_all_pct_all_clf_dummy = pd.concat(
        [df_all_pct_all_clf, df_dummy_join_with_all], axis=0, ignore_index=True, sort=False)
    # EXCLUDED BECAUSE THE PAIRWISE TESTING FAILS TO DISTINGUISH DUMMY VERSIONS FROM EACH OTHER
    # get_cd_diagram(df_all_pct_all_clf_dummy, df_all_pct_all_clf_ds_list,'accuracy',f'cd_accuracy_all_pct_all_clf_dummy')
    get_cd_diagram(df_all_pct_all_clf_dummy, df_all_pct_all_clf_ds_list,
                   'hm', f'cd_hm_all_pct_all_clf_dummy')

    ###### scatter plot ######
    # between w percenteges same classifier
    pct_combinations = list(itertools.combinations(pcts, 2))
    for pct1, pct2 in pct_combinations:
        for data_df, clf in clf_df_list:
            get_scatter_plot_same_clf_two_pct(
                data_df, pct1, pct2, clf, 'test_ds_score', f'{clf}{pct1}_vs_{clf}{pct2}_accuracy')
            get_scatter_plot_same_clf_two_pct(
                data_df, pct1, pct2, clf, 'harmonic_mean', f'{clf}{pct1}_vs_{clf}{pct2}_hm')

    export_data_for_recommendation(df_all_pct_all_clf)

    IPython.embed()

    ###### Training Time ######

    df_plots = df.copy()
    df_plots['length_100'] = df_plots['length']/100

    g = sns.FacetGrid(df, col="clf", hue="dim",
                      col_wrap=4,  margin_titles=True)
    g.map(sns.scatterplot, "test_ds_score", "num_dim", alpha=.7)
    g.add_legend()
    plt.savefig('abc')

    g = sns.FacetGrid(df[~df['dataset'].isin(
        ['InsectWingbeatSound', 'ElectricDevices'])], col="clf", hue="dim", col_wrap=4)
    g.map(sns.scatterplot, "train_size", "train_time", alpha=.7)
    g.add_legend()
    plt.savefig('abc')

    # duration by rev_pct
    g = sns.FacetGrid(df, col="classifier", col_wrap=2, height=2, ylim=(0, 10))
    g.map(sns.pointplot, "revealed_pct", "train_time", color=".3", ci=None)
    g.add_legend()
    plt.savefig('abc')

    # g = sns.FacetGrid(df, col="clf", hue="dim", col_wrap=4)
    # g.map(sns.scatterplot,"length", "train_time", alpha=.7)
    # g.add_legend()
    # g.set(yscale='log')
    # g.set(xscale='log')
    # plt.savefig('abc')

    # g = sns.FacetGrid(df, col="clf", hue="dim", col_wrap=4)
    # g.map(sns.regplot,"length", "train_time")
    # g.add_legend()
    # g.set(yscale='log')
    # g.set(xscale='log')
    # plt.savefig('abc')

    # include dummy for chunks
    # for i in classifiers:
    #     q = df_all_pct_all_clf[df_all_pct_all_clf['classifier']== i]
    #     w = q['dataset'].unique().tolist()
    #     df_dummy_join = filter_by_list(df_dummy, 'dataset', w)
    #     df_comb = pd.concat([q, df_dummy_join], axis=0, ignore_index=True, sort= False)
    #     get_cd_diagram(df_comb, w,'accuracy',f'36_cd_accuracy_all_pct_{i}_clf_dummy')
    #     get_cd_diagram(df_comb, w,'hm',f'36_cd_hm_all_pct_{i}_clf_dummy')

    # IPython.embed()

    ##
    df_all_cd_acc = get_cd_df_acc(df_all_pct_all_clf)
    df_all_cd_hm = get_cd_df_hm(df_all_pct_all_clf)

    # datasets = list(joined['dataset'].unique())
    # dataset_types = list(joined['type'].unique())
