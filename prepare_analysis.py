import os
import pandas as pd
import glob
import pathlib
from csv import reader

from pandas.io import json

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

def get_cd_df(df):
    # for CD diagram
    return df[['clf', 'dataset', 'test_ds_score']]

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
    with open('analysis_filenames', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            json_files.append(row)
    return json_files

def extract_json_data(json_files):
    dfs= []
    for j in json_files:
        df = pd.read_json(j)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True, sort= False)


pcts= [10, 20, 30, 100]
train_size = [50,100,250,500,1000]
length = [50,100,250,500,1000]
num_classes = [2,3,4,5,10,15,30,50]
analysis_filenames = 'log_filenames.csv'

json_files = get_json_files(analysis_filenames)
df = extract_json_data(json_files)
df = get_analysis_df(df)
df = create_clf_pct_col(df)

dataset_df = get_datasets_df()

joined = get_joined_df(df, dataset_df)

datasets = list(joined['dataset'].unique())
dataset_types = list(joined['type'].unique())

def rank_by_type(df):
    dataset_types = list(df['type'].unique())
    for type in dataset_types:



df_10 = filter_by_val(df, 'revealed_pct', 10)
df_20 = filter_by_val(df, 'revealed_pct', 20)
df_30 = filter_by_val(df, 'revealed_pct', 30)
df_100 = filter_by_val(df, 'revealed_pct', 100)


# get all datasets with revealed_pct and runs output
datasets = pd.read_csv('Datasets_metadata.csv')
revealed_pct = [10,20,30,100]
classifiers = ['ST', 'CBoss', 'TSF', 'PForest', 'WEASEL', 'Dummy']
datasets_extended = pd.concat([datasets['dataset']] * len(revealed_pct) , keys = revealed_pct).reset_index(level = 1, drop = True).rename_axis('revealed_pct').reset_index()
datasets_extended = pd.concat([datasets_extended] * len(classifiers) , keys = classifiers).reset_index(level = 1, drop = True).rename_axis('model').reset_index()

import IPython
IPython.embed()

# report = report.sort_values('ts').groupby(['model','dataset','revealed_pct']).tail(1)


# successful_runs = report[report['success']==True] 
# successful_runs.to_json('./successful_runs.json')

# # rerun = report[report['success']==False and ] 
# # rerun.to_json('./rerun.json')


# import IPython
# IPython.embed()

# # it should always be empty which means I didn't put failed dataset while it was successfully run after this failed run
# intersected_runs =successful_runs.merge(rerun, on=['model','dataset','revealed_pct'], how='inner')                                                              
# if not intersected_runs.empty:
#     print(f"check duplicated dataset runs{intersected_runs}")
