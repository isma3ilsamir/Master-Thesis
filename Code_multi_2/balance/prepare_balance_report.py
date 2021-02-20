import os
import pandas as pd

path_to_json = os.path.join(os.getcwd(), 'balance', '')
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]


dfs= []
for j in json_files:
    df = pd.read_json(os.path.join(path_to_json, j))
    dfs.append(df)

report = pd.concat(dfs, axis=0, ignore_index=True)

import IPython
IPython.embed()