import os
import pandas as pd
import glob

# path_to_json = os.path.join(os.getcwd(), 'logs', '')
path_to_json = glob.glob(os.path.join(os.getcwd(),'*/logs/*'))
json_files = [pos_json for pos_json in path_to_json if pos_json.endswith('.json')]



dfs= []
for j in json_files:
    df = pd.read_json(j)
    dfs.append(df)

report = pd.concat(dfs, axis=0, ignore_index=True)

report = report.sort_values('ts').groupby(['model','dataset','revealed_pct']).tail(1)
rerun = report[report['success']==False] 
rerun.to_json('./rerun.json')

successful_runs = report[report['success']==True] 
successful_runs.to_json('./successful_runs.json')
import IPython
IPython.embed()