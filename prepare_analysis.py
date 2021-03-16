import os
import pandas as pd
import glob

# path_to_json = os.path.join(os.getcwd(), 'logs', '')
path_to_json = glob.glob(os.path.join(os.getcwd(),'*/datasets/*/analysis/*'))
json_files = [pos_json for pos_json in path_to_json if pos_json.endswith('.json')]



dfs= []
for j in json_files:
    df = pd.read_json(j)
    dfs.append(df)

report = pd.concat(dfs, axis=0, ignore_index=True)

cols = ['classifier', 'mean_fit_time', 'std_fit_time', 'mean_score_time',
       'std_score_time', 'params', 'mean_test_score', 'std_test_score',
       'train_time', 'test_ds_score', 'test_ds_score_time',
       'revealed_pct', 'harmonic_mean', 'dataset']

# report = report.sort_values('ts').groupby(['model','dataset','revealed_pct']).tail(1)

import IPython
IPython.embed()
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
