import os
import pandas as pd
import glob
import pathlib

def create_clf_pct_col(df):
    df['clf'] = df['classifier'] + '_' + df['revealed_pct'].astype(str)
    return df

def get_joined_df(df, dataset_df):
    joined = df.set_index('dataset').join(dataset_df, how='inner')
    joined = joined.reset_index(drop = True)
    return joined

def filter_by_val(df, col, val):
    return df[df[col]== val]

def filter_by_list(df, col, values):
    return df[df[col].isin(values)]

def get_datasets_df(ds_file='Datasets_metadata.xlsx', index_col=0, sheet='Sheet3'):
    return pd.read_excel(ds_file, index_col= index_col, sheet_name= sheet)

def get_cd_df(df):
    # for CD diagram
    return df[['clf', 'dataset', 'test_ds_score']]

folders = [
 'code_cbos_wsl',
 'code_st',
 'code_tsf_inception',
 'pf_experiment',
 'pf_without_twe',
 'wsl_failed_rerun'
]
# path_to_json = os.path.join(os.getcwd(), 'logs', '')
path_to_json = []
for folder in folders:
    path_to_json.extend(glob.glob(os.path.join(os.getcwd(),f'{folder}/datasets/*/analysis/etsc*')))
json_files = [pos_json for pos_json in path_to_json if pos_json.endswith('.json')]

dfs= []
for j in json_files:
    df = pd.read_json(j)
    dfs.append(df)

report = pd.concat(dfs, axis=0, ignore_index=True, sort= False)

cols = ['classifier', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
       'std_score_time', 'params', 'mean_test_score', 'std_test_score',
       'train_time', 'test_ds_score', 'test_ds_score_time',
       'revealed_pct', 'harmonic_mean', 'dataset']

df = report[cols].copy()

cols_1 = ['classifier', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
       'std_score_time', 'mean_test_score', 'std_test_score',
       'train_time', 'test_ds_score', 'test_ds_score_time',
       'revealed_pct', 'harmonic_mean', 'dataset']
df_1 = report[cols_1].copy()
df_1.drop_duplicates(inplace= True)
unique_indexes = df_1.index

df= df.filter(items= unique_indexes, axis= 'index')
df.reset_index(inplace= True)
dataset_df = get_datasets_df()

joined = get_joined_df(df, dataset_df)
pcts= [10, 20, 30, 100]

df_10 = filter_by_val(df, 'revealed_pct', 10)
df_20 = filter_by_val(df, 'revealed_pct', 20)
df_30 = filter_by_val(df, 'revealed_pct', 30)
df_100 = filter_by_val(df, 'revealed_pct', 100)


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
