import os
import pandas as pd
import glob
import pathlib
from csv import reader
from pandas.io import json

def round_down(num, divisor):
    return num - (num%divisor)

def create_clf_pct_col(df):
    df['clf'] = df['classifier'] + '_' + df['revealed_pct'].astype(str)
    return df

def get_joined_df(df, dataset_df):
    joined = df.set_index('dataset').join(dataset_df.set_index('dataset'), how='inner')
    joined = joined.reset_index(drop = True)
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
            analysis_path = glob.glob(os.path.join(os.getcwd(),f'*/datasets/*/analysis/{file_name}'))
            if not analysis_path:
                raise(f'there was no analysis file found for {row[0]}')
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

def get_cd_df(df):
    # for CD diagram
    cd_df = df['clf','dataset','test_ds_score']
    cd_df.columns = ['classifier_name','dataset_name','accuracy']
    return cd_df

def rank_by_type(df):
    dataset_types = list(df['type'].unique())
    for type in dataset_types:
        pass
    return False

def get_runs_within_clf(df, classifier, dataset, revealed_pcts):
    df.query(f'classifier == {classifier} &\
               dataset == {dataset} &\
               FT_Team.str.startswith("S").values')

    return False

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

pcts= [10, 20, 30, 100]
train_size = [50,100,250,500,1000]
length = [50,100,250,500,1000]
num_classes = [2,3,4,5,10,15,30,50]
classifiers = ['ST', 'CBoss', 'TSF', 'PForest', 'WEASEL', 'Dummy']
analysis_filenames = 'log_filenames.csv'

json_files = get_json_files(analysis_filenames)
df = extract_json_data(json_files)
df = get_analysis_df(df)
df = create_clf_pct_col(df)

dataset_df = get_datasets_df()
dataset_extended = get_extended_datasets(dataset_df, pcts, classifiers)

import IPython
IPython.embed()

joined = get_joined_df(df, dataset_df)

datasets = list(joined['dataset'].unique())
dataset_types = list(joined['type'].unique())

df_10 = filter_by_val(df, 'revealed_pct', 10)
df_20 = filter_by_val(df, 'revealed_pct', 20)
df_30 = filter_by_val(df, 'revealed_pct', 30)
df_100 = filter_by_val(df, 'revealed_pct', 100)