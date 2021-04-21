import os
import pandas as pd
import glob
import pathlib
from csv import reader
from pandas.io import json
import IPython
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def round_down(num, divisor):
    return num - (num%divisor)

def get_error(df):
    df_copy = df.copy()
    df_copy['error_rate'] = 1 - df_copy['test_ds_score']
    return df_copy

def get_rev_hm(df):
    df_copy = df.copy()
    df_copy['reversed_hm'] = (2 * (df_copy['revealed_pct_actual']/100) * df_copy['error_rate']) / ((df_copy['revealed_pct_actual']/100) + df_copy['error_rate'])
    return df_copy

def adjust_hm(df):
    df_copy = df.copy()
    df_copy['harmonic_mean'] = 1- df_copy['reversed_hm']
    return df_copy

# def adjust_hm(df):
#     df_copy = df.copy()
#     df_copy['harmonic_mean'] = (2 * (1 - (df_copy['revealed_pct_actual']/100)) * df_copy['test_ds_score']) / ((1 - (df_copy['revealed_pct_actual']/100)) + df_copy['test_ds_score'])
#     return df_copy

def create_whole_revpct(df):
    df['revealed_pct_actual'] = df['revealed_pct']
    # round revealed_pct
    df['revealed_pct'] = round_down(df['revealed_pct_actual'],10)
    return df

def create_clf_pct_col(df):
    df = create_whole_revpct(df)
    df['clf'] = df['classifier'] + '_' + df['revealed_pct'].astype(str)
    return df

def get_joined_df(df, dataset_df):
    joined = df.set_index('dataset').join(dataset_df.set_index('dataset'), how='inner')
    joined = joined.reset_index(drop = False)
    return joined

def filter_by_val(df, col, val):
    return df[df[col]== val]

def filter_by_list(df, col, values):
    return df[df[col].isin(values)]

def get_datasets_df(ds_file='Datasets_metadata.xlsx', index_col=0, sheet='Sheet3'):
    # return pd.read_excel(ds_file, index_col= index_col, sheet_name= sheet)
    return pd.read_csv('Datasets_metadata.csv')

def get_analysis_df(df):
    cols = ['classifier', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
       'std_score_time', 'params', 'mean_test_score', 'std_test_score',
       'train_time', 'test_ds_score', 'test_ds_score_time',
       'revealed_pct', 'harmonic_mean', 'dataset']
    return df[cols].copy()

def remove_duplicates(df):
    cols = df.columns.drop('params')
    df_1 = df[cols].copy()
    df_1.drop_duplicates(inplace= True)
    unique_indexes = df_1.index
    df= df.filter(items= unique_indexes, axis= 'index')
    df.reset_index(inplace= True)
    return df

def get_json_files(analysis_filenames):
    json_files = []
    with open(analysis_filenames, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[0] == 'filename':
                continue
            file_name = os.path.basename(row[0])
            # analysis_path = glob.glob(os.path.join(os.getcwd(),f'*/datasets/*/analysis/{file_name}'))
            analysis_path = glob.glob(os.path.join(os.getcwd(),f'*/{file_name}'))
            if not analysis_path:
                IPython.embed()
                raise Exception(f'there was no analysis file found for {row[0]}')
            elif len(analysis_path) > 1:
                print(f'there is more than one location for file {row[0]}:')
                print(analysis_path)
            json_files.append(analysis_path[0])
    return json_files

def extract_json_data(json_files):
    dfs= []
    for j in json_files:
        df = pd.read_json(j)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True, sort= False)

def get_cd_df_acc(df):
    # for CD diagram
    cd_df = df[['clf','dataset','test_ds_score']]
    cd_df.columns = ['classifier_name','dataset_name','accuracy']
    return cd_df

def get_cd_df_hm(df):
    # for CD diagram
    cd_df = df[['clf','dataset','harmonic_mean']]
    cd_df.columns = ['classifier_name','dataset_name','accuracy']
    return cd_df

def rank_on_ds_score(df):
    dfs=[]
    datasets = list(df['dataset'].unique())
    for ds in datasets:
        filtered= filter_by_val(df,'dataset', ds).copy()
        filtered['ds_rank'] = filtered['test_ds_score'].rank(ascending=False, method='min')
        dfs.append(filtered)
    ranked = pd.concat(dfs, axis=0, ignore_index=True)
    return ranked

def get_used_train_length(row, splits=10):
    length = row['length']
    pct = row['revealed_pct']
    split_lengths= ([length // splits + (1 if x < length % splits else 0)  for x in range (splits)])
    split_indexes= np.cumsum(split_lengths).tolist()
    return split_indexes[int(pct/10) - 1]

def get_standard_train_length(row):
    pct10_length = [5,10,25,50,100]
    pct20_length = [10,20,50,100,200]
    pct30_length = [15,30,75,150,300]
    pct100_length = [50,100,250,500,1000]
    pct = row['revealed_pct']
    if pct == 10:
        train_length_std = apply_metric_standardization(row, pct10_length, 'train_length')
    elif pct == 20:
        train_length_std = apply_metric_standardization(row, pct20_length, 'train_length')
    elif pct == 30:
        train_length_std = apply_metric_standardization(row, pct30_length, 'train_length')
    elif pct == 100:
        train_length_std = apply_metric_standardization(row, pct100_length, 'train_length')
    else:
        raise Exception(f"No handling for {pct}% in length std")
    return train_length_std

def get_standard_train_size(row):
    sizes = [50,100,250,500,1000]
    train_size_std = apply_metric_standardization(row, sizes, 'train_size')
    return train_size_std

def get_standard_num_classes(row):
    classes = [2,3,4,5,10,15,30,50]
    train_size_std = apply_metric_standardization(row, classes, 'num_classes')
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

        if (row[col] > min) and (row[col] <=max):
            if max != np.inf:
                return f"{min}-{max}"
            else:
                return f'{min}+'
        else:
            continue

def get_ds_finished_chunks_for_clf(df, classifier, num_chunks):
    df = filter_by_val(df, 'classifier', classifier)
    ds_chunks = df.groupby('dataset')['revealed_pct'].size().to_frame().reset_index()
    ds_chunks.columns = ['dataset','chunks_finished']
    ds_list = filter_by_val(ds_chunks, 'chunks_finished', num_chunks)['dataset'].tolist()
    return ds_list

def get_ds_finished_clfs_for_revpct(df, revealed_pct, num_classifiers):
    df = filter_by_val(df, 'revealed_pct', revealed_pct)
    ds_clfs = df.groupby('dataset')['classifier'].size().to_frame().reset_index()
    ds_clfs.columns = ['dataset','classifiers_finished']
    ds_list = filter_by_val(ds_clfs, 'classifiers_finished', num_classifiers)['dataset'].tolist()
    return ds_list

def get_extended_datasets(dataset_df, revealed_pct, classifiers):
    # get all datasets with revealed_pct and runs output
    datasets_extended = pd.concat([dataset_df['dataset']] * len(revealed_pct) , keys = revealed_pct).reset_index(level = 1, drop = True).rename_axis('revealed_pct').reset_index()
    datasets_extended = pd.concat([datasets_extended] * len(classifiers) , keys = classifiers).reset_index(level = 1, drop = True).rename_axis('model').reset_index()
    return datasets_extended

def get_extended_dataset_df(dataset_df, pcts, classifiers):
    dataset_extended = get_extended_datasets(dataset_df, pcts, classifiers)
    return dataset_extended

def get_main_df(dataset_df):
    json_files = get_json_files(analysis_filenames)
    df = extract_json_data(json_files)
    df = get_analysis_df(df)
    df = create_clf_pct_col(df)
    df = get_joined_df(df, dataset_df)
    df['train_length'] = df.apply(get_used_train_length, axis=1)
    df['train_length_std'] = df.apply(get_standard_train_length, axis=1)
    df['train_size_std'] = df.apply(get_standard_train_size, axis=1)
    df['num_classes_std'] = df.apply(get_standard_num_classes, axis=1)
    df = get_error(df)
    df = get_rev_hm(df)
    df = adjust_hm(df)
    return df



if __name__ == "__main__":
    pcts= [10, 20, 30, 100]
    classifiers = ['ST', 'CBoss', 'TSF', 'PForest', 'WEASEL', 'Dummy']
    analysis_filenames = 'log_filenames.csv'
    dataset_df = get_datasets_df()
    dataset_extended = get_extended_dataset_df(dataset_df, pcts, classifiers)
    df = get_main_df(dataset_df)

    ###### across clf analysis ######
    ## 10% datasets
    ds_10pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df, 10, 5)
    ds_10pct_data = filter_by_val(df, 'revealed_pct', 10)
    df_10pct = filter_by_list(ds_10pct_data, 'dataset', ds_10pct_finished_5clf_list)

    ## 20% datasets
    ds_20pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df, 20, 5)
    ds_20pct_data = filter_by_val(df, 'revealed_pct', 20)
    df_20pct = filter_by_list(ds_20pct_data, 'dataset', ds_20pct_finished_5clf_list)

    ## 30% datasets
    ds_30pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df, 30, 5)
    ds_30pct_data = filter_by_val(df, 'revealed_pct', 30)
    df_30pct = filter_by_list(ds_30pct_data, 'dataset', ds_30pct_finished_5clf_list)

    ## 100% datasets
    ds_100pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df, 100, 5)
    ds_100pct_data = filter_by_val(df, 'revealed_pct', 100)
    df_100pct = filter_by_list(ds_100pct_data, 'dataset', ds_100pct_finished_5clf_list)


    # all datasets all classifiers (excluding Dummy and 100 pct classifiers)
    df1 = df[df['classifier']!='Dummy']
    ds1_10pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df1, 10, 5)
    ds1_20pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df1, 20, 5)
    ds1_30pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df1, 30, 5)
    ds1_100pct_finished_5clf_list = get_ds_finished_clfs_for_revpct(df1, 100, 5)
    ds_all_clf_all_pct = set(ds1_10pct_finished_5clf_list).intersection(set(ds1_20pct_finished_5clf_list),set(ds1_30pct_finished_5clf_list),set(ds1_100pct_finished_5clf_list))
    # ds_all_clf_all_pct = set(ds1_10pct_finished_5clf_list).intersection(set(ds1_20pct_finished_5clf_list),set(ds1_30pct_finished_5clf_list))
    df_all = filter_by_list(df1, 'dataset', ds_all_clf_all_pct)
    # df_all = df_all[df_all['revealed_pct']!= 100]
    df_all_cd_acc = get_cd_df_acc(df_all)
    df_all_cd_hm = get_cd_df_hm(df_all)

    ## CD same pct revealed
    ds_10pct_cd_acc = get_cd_df_acc(df_10pct)
    ds_10pct_cd_hm = get_cd_df_hm(df_10pct)
    # ds_10pct_cd.to_csv('pforest_example.csv', index=False)

    ## rank on data set
    df_10_pct_ranked = rank_on_ds_score(df_10pct)
    df_10_pct_ranked_1_only = df_10_pct_ranked[df_10_pct_ranked['ds_rank']==1]
    ### by type
    df_10_pct_ranked_type = pd.pivot_table(df_10_pct_ranked_1_only, values='ds_rank', columns=['classifier'], aggfunc=np.sum, index=['type'])
    # df_10_pct_ranked_type.to_csv('ranking.csv', index=False)
    ### by length
    df_10_pct_ranked_length = pd.pivot_table(df_10_pct_ranked_1_only, values='ds_rank', columns=['classifier'], aggfunc=np.sum, index=['train_length_std'])
    # df_10_pct_ranked_length.to_csv('ranking.csv', index=False)
    ### by train size
    df_10_pct_ranked_size = pd.pivot_table(df_10_pct_ranked_1_only, values='ds_rank', columns=['classifier'], aggfunc=np.sum, index=['train_size_std'])
    # df_10_pct_ranked_length.to_csv('ranking.csv', index=False)
    ### by num classes
    df_10_pct_ranked_size = pd.pivot_table(df_10_pct_ranked_1_only, values='ds_rank', columns=['classifier'], aggfunc=np.sum, index=['num_classes_std'])
    # df_10_pct_ranked_length.to_csv('ranking.csv', index=False)

    ###### within clf analysis ######

    ## TSF
    tsf_finished_4chunks_list = get_ds_finished_chunks_for_clf(df, 'TSF', 4)
    tsf_data = filter_by_val(df, 'classifier', 'TSF')
    tsf_4chunks = filter_by_list(tsf_data, 'dataset', tsf_finished_4chunks_list)

    ## CD between diff pct revealed
    tsf_4chunks_cd_acc = get_cd_df_acc(tsf_4chunks)
    tsf_4chunks_cd_hm = get_cd_df_hm(tsf_4chunks)
    # tsf_4_chunks_cd.to_csv('pforest_example.csv', index=False)

    ## scatter tsf100 vs tsf_10
    tsf100_vs_tsf10= pd.pivot_table(tsf_4chunks, values='test_ds_score', columns=['revealed_pct'], aggfunc=np.sum, index=['dataset'])
    plt.plot(tsf100_vs_tsf10[10], tsf100_vs_tsf10[100], 'o', color='black')
    # plt.plot([0,1], [0, 1], 'k-')
    plt.xlabel('TSF_10', fontsize=16)
    # plt.margins(x=0)
    plt.ylabel('TSF_100', fontsize=16)
    # plt.margins(y=0)
    plt.fill([0,0,1,0], [0,1,1,0], 'lightskyblue', alpha=0.2, edgecolor='lightskyblue')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plt.savefig('TSF100_vs_TSF10.jpg', dpi=200)
    plt.show()


    import IPython
    IPython.embed()

    # datasets = list(joined['dataset'].unique())
    # dataset_types = list(joined['type'].unique())

    # df_10 = filter_by_val(df, 'revealed_pct', 10)
    # df_20 = filter_by_val(df, 'revealed_pct', 20)
    # df_30 = filter_by_val(df, 'revealed_pct', 30)
    # df_100 = filter_by_val(df, 'revealed_pct', 100)