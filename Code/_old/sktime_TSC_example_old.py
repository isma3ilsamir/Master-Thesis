from time import process_time
import numpy as np

from splitting import split_ts_5_pct
from datasets import get_test_train_data, lookup_dataset

from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

from sktime.transformers.series_as_features.compose import ColumnConcatenator
# from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
# from sktime.classification.dictionary_based import BOSSEnsemble
# from sklearn.metrics import accuracy_score




if __name__ == '__main__':
    X_train, X_test, y_train, y_test= get_test_train_data('BasicMotions')
    X_train_1, split_indexes= split_ts_5_pct(X_train, desc= True)

    steps = [
    ('concatenate', ColumnConcatenator()),
    ('classify', TimeSeriesForestClassifier(n_estimators=100))]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    for i in range(20,0,-1):
        X_train_2= X_train_1[[f'{5*i}_pct_dim_0',f'{5*i}_pct_dim_1']]
        steps = [
        ('concatenate', ColumnConcatenator()),
        ('classify', TimeSeriesForestClassifier(n_estimators=100))]
        clf = Pipeline(steps)
        clf.fit(X_train_2, y_train)
        print(f'Score of {5*i}% ts training classifier= {clf.score(X_test, y_test)}')

    # clf = ColumnEnsembleClassifier(estimators=[
    # ('TSF0', TimeSeriesForestClassifier(n_estimators=100), [0]),
    # ('BOSSEnsemble3', BOSSEnsemble(max_ensemble_size=5), [3]),])
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # labels, counts = np.unique(y_train, return_counts= True)
    # fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    # for label in labels:
    #     X_train.loc[y_train == label, "dim_0"].iloc[1].plot(ax=ax, label=f"class {label}")
    # plt.legend()
    # ax.set(title="Example time series", xlabel="Time")

    # labels, counts = np.unique(y_train, return_counts= True)
    # fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    # for label in labels:
    #     X_train_1.loc[y_train == label, "20splitdim_0"].iloc[1].plot(ax=ax, label=f"class {label}")
    # plt.legend()
    # ax.set(title="Example time series", xlabel="Time")