from time import process_time
import numpy as np

from splitting import split_ts_5_pct, get_split_indexes, apply_split
from datasets import get_test_train_data, lookup_dataset

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
import matplotlib.pyplot as plt

from sktime.classification.distance_based._time_series_neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based._elastic_ensemble import ElasticEnsemble
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import BOSSIndividual

# from sktime.classification.compose import ColumnEnsembleClassifier
# from sktime.transformers.series_as_features.compose import ColumnConcatenator

from tslearn.utils import from_sktime_dataset, to_sktime_dataset
from tslearn.utils import to_pyts_dataset, from_pyts_dataset

from pyts.classification import KNeighborsClassifier
from pyts.classification import LearningShapelets


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_test_train_data('GunPoint')
    split_indexes = get_split_indexes(X_train)

    # clf = TimeSeriesForestClassifier()
    # train_start= process_time()
    # clf.fit(X_train, y_train)
    # predict_start= process_time()
    # y_pred = clf.predict(X_test)
    # predict_stop= process_time()
    # print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    # print(classification_report(y_test, y_pred))
    # print(clf.score(X_test, y_test))

    # params = {'n_estimators': [1, 10, 100, 200, 500],
    #           'max_features': ['sqrt', 'log2', None],
    #           'n_jobs': [-1]}
######################################################################
#       Doing grid search for same data but different TS length
######################################################################
    # print("grid searches")
    # Grids={}
    # for i, index in enumerate(split_indexes):
    #     X_train_2 = apply_split(X_train, index)
    #     gs = GridSearchCV(clf,params)
    #     gs.fit(X_train_2, y_train)
    #     Grids[f'{5*(i+1)}_pct']=gs

    # print("default model")
    # for i, index in enumerate(split_indexes):
    #     X_train_2 = apply_split(X_train, index)
    #     steps = [
    #         # ('concatenate', ColumnConcatenator()),
    #         ('classify', TimeSeriesForestClassifier())]
    #     clf = Pipeline(steps)
    #     clf.fit(X_train_2, y_train)
    #     print(
    #         f'Score of {5*i}% ts training classifier= {clf.score(X_test, y_test)}')

    # print("model with random parameters")
    # for i, index in enumerate(split_indexes):
    #     X_train_2 = apply_split(X_train, index)
    #     steps = [
    #         # ('concatenate', ColumnConcatenator()),
    #         ('classify', TimeSeriesForestClassifier(n_estimators=500,
    #                                                 criterion="entropy",
    #                                                 bootstrap=True,
    #                                                 oob_score=True,
    #                                                 random_state=1,
    #                                                 n_jobs=-1,))]
    #     clf = Pipeline(steps)
    #     clf.fit(X_train_2, y_train)
    #     print(
    #         f'Score of {5*i}% ts training classifier= {clf.score(X_test, y_test)}')

    # print("model from grid search")
    # for i, index in enumerate(split_indexes):
    #     X_train_2 = apply_split(X_train, index)
    #     steps = [
    #         # ('concatenate', ColumnConcatenator()),
    #         ('classify', TimeSeriesForestClassifier(n_estimators=100,
    #                                                 criterion="entropy",
    #                                                 bootstrap=True,
    #                                                 max_features= 'log2',
    #                                                 oob_score=True,
    #                                                 random_state=1,
    #                                                 n_jobs=-1,))]
    #     clf = Pipeline(steps)
    #     clf.fit(X_train_2, y_train)
    #     print(
    #         f'Score of {5*i}% ts training classifier= {clf.score(X_test, y_test)}')
######################################################################
#       Testing the grid search
######################################################################
#     X_train_2 = apply_split(X_train, 40)
#     clf = TimeSeriesForestClassifier(max_features='log2',
#                                      n_estimators=100,
#                                      n_jobs=-1)
#     clf.fit(X_train_2, y_train)
#     predict_start= process_time()
#     y_pred = clf.predict(X_test)
#     predict_stop= process_time()
#     print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
#     print("results for ts length 25%")
#     print(classification_report(y_test, y_pred))

#     X_train_2 = apply_split(X_train, 80)
#     clf = TimeSeriesForestClassifier(max_features=None,
#                                      n_estimators=100,
#                                      n_jobs=-1)
#     clf.fit(X_train_2, y_train)
#     predict_start= process_time()
#     y_pred = clf.predict(X_test)
#     predict_stop= process_time()
#     print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
#     print("results for ts length 50%")
#     print(classification_report(y_test, y_pred))

#     X_train_2 = apply_split(X_train, 122)
#     clf = TimeSeriesForestClassifier(max_features='sqrt',
#                                      n_estimators=100,
#                                      n_jobs=-1)
#     clf.fit(X_train_2, y_train)
#     predict_start= process_time()
#     y_pred = clf.predict(X_test)
#     predict_stop= process_time()
#     print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
#     print("results for ts length 80%")
#     print(classification_report(y_test, y_pred))

# ######################################################################
# #       Doing grid search for same data but different TS length
# ######################################################################
#     clf = TimeSeriesForest()
#     train_start= process_time()
#     clf.fit(X_train, y_train)
#     predict_start= process_time()
#     y_pred = clf.predict(X_test)
#     predict_stop= process_time()
#     print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
#     print(classification_report(y_test, y_pred))

#     params = {'n_estimators': [1, 5, 10, 20, 50, 80, 100, 150, 200, 500],
#               #   'max_features': ['sqrt', 'log2', None],
#               'n_jobs': [-1]}

#     gs = GridSearchCV(clf, params)
#     gs.fit(X_train, y_train)
# ######################################################################
# #       trying cross val
# ######################################################################
#     cvs = cross_val_score(clf, X_train, y_train, cv=5)

#     y_pred = gs.best_estimator_.predict(X_test)
#     print(classification_report(y_test, y_pred))


######################################################################
#       1NNED
######################################################################
    x_tsl = from_sktime_dataset(X_train)
    x = to_pyts_dataset(x_tsl)
    clf = KNeighborsClassifier(weights='uniform', algorithm='brute',
                               metric='minkowski',
                               p=2)
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(x, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    x_tst_tsl = x_tsl = from_sktime_dataset(X_test)
    x_tst = to_pyts_dataset(x_tst_tsl)
    predict_start= process_time()
    y_prd = clf.predict(x_tst)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    clf.score(x_tst, y_test)
    print(classification_report(y_test, y_prd))
######################################################################
#       1NNDTW
######################################################################
    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, weights='uniform', algorithm='brute', metric='dtw')
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    clf.score(X_test, y_test)
    print(classification_report(y_test, y_pred))


######################################################################
#       Elastic Ensemble -not working-
######################################################################
    # clf = ElasticEnsemble(distance_measures='all',
    #                       proportion_of_param_options=1,
    #                       proportion_train_for_test=1,
    #                       proportion_train_in_param_finding=1,
    #                       verbose=1)
    # train_start= process_time()
    # clf.fit(X_train, y_train)
######################################################################
#       1NNMSM
######################################################################
    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, weights='uniform', algorithm='brute', metric='msm')
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    clf.score(X_test, y_test)
    print(classification_report(y_test, y_pred))


######################################################################
#       Timeseries Forest
######################################################################
    clf = TimeSeriesForestClassifier()
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))
######################################################################
#       Time Series Bag of Features (TSBF) -not found-
######################################################################


######################################################################
#       Learned Shapelets
######################################################################
    x_tsl = from_sktime_dataset(X_train)
    x = to_pyts_dataset(x_tsl)
    clf = LearningShapelets(random_state=42, tol=0.01)
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(x, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    x_tst_tsl = x_tsl = from_sktime_dataset(X_test)
    x_tst = to_pyts_dataset(x_tst_tsl)
    predict_start= process_time()
    y_prd = clf.predict(x_tst)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    clf.score(x_tst, y_test)
    print(classification_report(y_test, y_prd))
######################################################################
#       Transformed Shapelets
######################################################################
    clf = ShapeletTransformClassifier(time_contract_in_mins=2)
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))


######################################################################
#       WEASEL
######################################################################
    clf = WEASEL(anova=True,
                 bigrams=True,
                 binning_strategy='information-gain',
                 window_inc=4, chi2_threshold=-1)
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))
######################################################################
#       BOSS Individual
######################################################################
    clf= BOSSIndividual(window_size= 10,
    word_length= 8,
    norm= False,
    alphabet_size= 4,
    save_words= True)
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))
######################################################################
#       BOSS Ensemble
######################################################################
    clf= BOSSEnsemble()
    clf_name= clf.__module__.split('.')[-1]
    train_start= process_time()
    clf.fit(X_train, y_train)
    train_stop= process_time()
    print(f'Training Time of {clf_name} in seconds:', train_stop-train_start)
    predict_start= process_time()
    y_pred = clf.predict(X_test)
    predict_stop= process_time()
    print(f'Prediction Time of {clf_name} in seconds:', predict_stop-predict_start)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))
