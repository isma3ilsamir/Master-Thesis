import os
import pandas as pd
import glob
import csv

# path_to_json = os.path.join(os.getcwd(), 'logs', '')
path_to_json = glob.glob(os.path.join(os.getcwd(),'*/logs/*'))
json_files = [pos_json for pos_json in path_to_json if pos_json.endswith('.json')]


dfs= []
for j in json_files:
    df = pd.read_json(j)
    df['filename'] = j
    dfs.append(df)

report = pd.concat(dfs, axis=0, ignore_index=True)
# format date correctly 
report['ts'] =pd.to_datetime(report['ts'], format='%d-%m-%Y %H:%M:%S')  
# remove duplicates
# report.drop_duplicates(inplace= True)
# filter on the last trials for experiment run
filtered = report[report['ts']>'22-02-2021']

# get last combination of dataset with model
filtered = filtered.sort_values('ts').groupby(['model','dataset','revealed_pct']).tail(1)

successful_runs = filtered[filtered['success']==True] 
successful_runs.to_json('./successful_runs.json')

rerun = filtered[filtered['success']==False ] 
rerun.to_json('./rerun.json')

# run checks
results = []
# 1.check that each dataset has 4 runs against each model
results.append(successful_runs.groupby(['model','dataset']).size())
# 2.check that the number of dataset that run for each classifier 5 clf (336), dummy (84), MSM& LS (4)
results.append(successful_runs.groupby('model').size())

# 3.check totals 1680 - 5 classifiers 84 - Dummy 4 - MSM & LS
results.append(successful_runs.count())

import IPython
# IPython.embed()
# generate output list of valid files to be used for analysis
header = ['filename']
file_names = successful_runs['filename'].tolist()
with open('log_filenames.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(header)
    for item in file_names:
        write.writerow([item,])

# print report result
report_result = pd.concat(results, axis=0, ignore_index=False)
report_result.to_json('./reportresult.json')

# it should always be empty which means I didn't put failed dataset while it was successfully run after this failed run
intersected_runs =successful_runs.merge(rerun, on=['model','dataset','revealed_pct'], how='inner')                                                              
if not intersected_runs.empty:
    print(f"check duplicated dataset runs{intersected_runs}")

