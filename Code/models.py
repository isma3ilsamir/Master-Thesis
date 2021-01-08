from time import process_time
import os
import logging
import pandas as pd
import joblib

# from scipy.sparse.construct import rand
# from scipy.stats.stats import _euclidean_dist
from scipy.spatial.distance import euclidean
from scipy.stats.stats import mode
# from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.utils import estimator_checks
# from sklearn.utils.fixes import loguniform
# from scipy.stats import uniform, randint, poisson, norm, logser
from tslearn.utils import from_sktime_dataset, to_pyts_dataset

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.distance_based._time_series_neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.distance_based import ProximityForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.dictionary_based import ContractableBOSS

from pyts.classification import KNeighborsClassifier
from pyts.classification.learning_shapelets import LearningShapelets
from pyts.multivariate.classification import MultivariateClassifier

from sktime_dl.deeplearning import InceptionTimeClassifier

import seaborn as sns
import matplotlib.pyplot as plt
# from random import choices


class Model:
    clf = None
    clf_name = None
    hyper_param = {}
    cv = None
    models_folder = None
    logger = None

    def __init__(self, scoring_function, n_iter, cv, dim, ds):
        self.scoring_function = scoring_function
        self.n_iter = n_iter
        self.cv = cv
        self.initialize_logger()
        self.dim = dim
        self.initialize_model(ds)
        self.logger.info(f'{self.clf_name}: Initialization complete')

    def fit(self, X_train, y_train):
        self.logger.info(f'{self.clf_name}: Training started')
        if self.cv:
            self.fit_cv(X_train, y_train)
        else:
            self.fit_no_cv(X_train, y_train)
        self.logger.info(f'{self.clf_name}: Training Finished')

    def fit_no_cv(self, X_train, y_train):
        train_start = process_time()
        self.clf.fit(X=X_train, y=y_train)
        train_stop = process_time()
        self.train_time = train_stop - train_start

    def fit_cv(self, X_train, y_train):
        search = RandomizedSearchCV(
            estimator=self.clf,
            param_distributions=self.hyper_param,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=0,
            cv=5
        )
        search.fit(X=X_train, y=y_train)
        self.cv_obj = search
        self.cv_result = pd.DataFrame(search.cv_results_)
        self.best_estimator_ = search.best_estimator_
        self.best_params = search.best_params_
        self.best_estimator_analysis = self.get_best_estimator_analysis()
        self.train_time = self.best_estimator_analysis['mean_fit_time']

    def predict(self, X_test, y_test=None):
        self.logger.info(f'{self.clf_name}: Testing started')
        predict_start = process_time()
        if self.cv:
            y_pred = self.best_estimator_.predict(X=X_test)
        else:
            y_pred = self.clf.predict(X=X_test)
        predict_stop = process_time()
        self.predict_time = predict_stop - predict_start
        self.y_pred = y_pred
        self.logger.info(f'{self.clf_name}: Testing Finished')
        # return y_pred

    def predict_proba(self, X_test, y_test=None):
        self.logger.info(f'{self.clf_name}: Predicting Class Probabilities')
        if self.cv:
            proba = self.best_estimator_.predict_proba(X=X_test)
        else:
            proba = self.clf.predict_proba(X=X_test)
        self.proba = proba
        self.logger.info(f'{self.clf_name}: Class Probabilities Finished')
        # return proba

    def get_score(self, X_test, y_test):
        self.logger.info(f'{self.clf_name}: Calculating Accuracy Score')
        score_start = process_time()
        if self.cv:
            score = self.best_estimator_.score(X=X_test, y=y_test)
        else:
            score = self.clf.score(X=X_test, y=y_test)
        score_stop = process_time()
        self.test_ds_score_time = score_stop - score_start
        self.test_ds_score = score
        self.logger.info(f'{self.clf_name}: Accuracy Finished')
        # return score

    def get_best_estimator_analysis(self):
        name = pd.Series({'classifier': self.clf_name})
        analysis = self.cv_result.iloc[self.cv_obj.best_index_]
        return pd.concat([name, analysis], axis=0)

    def initialize_logger(self):
        # create logger
        self.logger = logging.getLogger(self.clf_name)
        self.logger.setLevel(logging.INFO)

        # create console handler and set level to debug
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # add formatter to ch
            ch.setFormatter(formatter)
            # add ch to logger
            self.logger.addHandler(ch)

    def initialize_model(self, ds):
        models_folder = os.path.join(os.getcwd(), 'datasets', ds, 'models\\')
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.models_folder = models_folder
        if self.dim > 1:
            mv_estimators, mv_param_dict = self.get_params_dict()
            self.clf = ColumnEnsembleClassifier(estimators=mv_estimators)
            self.hyper_param = mv_param_dict
        else:
            pass

    def get_params_dict(self):
        estimators = []
        params_dict = None
        for i in range(self.dim):
            temp = f"{self.clf_name}_{i}__"

            estimators.append((
                f"{self.clf_name}_{i}",
                self.clf,
                [i]))

            params_dict = {temp + str(key): val for key,
                           val in self.hyper_param.items()}
        return estimators, params_dict

    def export_model(self, exp):
        cv = 'cv' if self.cv else 'no_cv'
        fname = os.path.join(self.models_folder,
                             f'{exp}_{self.clf_name}_{cv}.sav')
        joblib.dump(self.clf, fname)
        self.logger.info(f'Model exported to {fname}')


class pytsModel(Model):

    def from_sktime_to_pyts(self, ds):
        self.logger.info(f'{self.clf_name}: Transforming Data to Pyts Format')
        ds_tsl = from_sktime_dataset(ds)
        ds_pyts = to_pyts_dataset(ds_tsl)
        return ds_pyts

    def fit(self, X, y):
        X_train = self.from_sktime_to_pyts(X)
        y_train = y
        Model.fit(self, X_train, y_train)

    def predict(self, X, y=None):
        X_test = self.from_sktime_to_pyts(X)
        y_test = y
        Model.predict(self, X_test, y_test)
        # return y_pred

    def predict_proba(self, X, y=None):
        X_test = self.from_sktime_to_pyts(X)
        y_test = y
        Model.predict(self, X_test, y_test)
        # return proba

    def get_score(self, X, y):
        X_test = self.from_sktime_to_pyts(X)
        y_test = y
        Model.get_score(self, X_test, y_test)
        # return score

    def initialize_model(self, ds):
        models_folder = os.path.join(os.getcwd(), 'datasets', ds, 'models\\')
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.models_folder = models_folder
        if self.dim > 1:
            self.hyper_param = self.get_params_dict()
            self.clf = MultivariateClassifier(self.clf)
        else:
            pass

    def get_params_dict(self):
        params_dict = None
        for i in range(self.dim):
            temp = f"estimator__"
            params_dict = {temp + str(key): val for key,
                           val in self.hyper_param.items()}
        return params_dict


class KNNED(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='minkowski',
                               p=2)
    clf_name = "1NN-ED"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        "leaf_size": [1, 10, 30, 100]
    }


class KNNED_SKTIME(Model):
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1,
                                         metric=euclidean)
    clf_name = "1NN-ED-SKTime"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
    }


class KNNDTW(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='dtw')
    clf_name = "1NN-DTW"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        "leaf_size": [1, 10, 30, 100]
    }


class KNNMSM(Model):
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='msm')
    clf_name = "1NN-MSM"
    hyper_param = {
        'algorithm': ['brute'],
        'weights': ['uniform', 'distance'],
        'metric_params': [{'c': 0.01},{'c': 0.1},{'c': 1},{'c': 10},{'c': 100}]
    }

class EE(Model):
    clf= ElasticEnsemble(verbose=0)
    clf_name= 'EE'
    hyper_param = {
        'distance_measures': ['all']
    }

class PFOREST(Model):
    clf = ProximityForest(n_jobs= -1,
    get_distance_measure= None,
    max_depth= 10,
    verbosity= 3)
    clf_name = 'PForest'
    hyper_param = {
        'n_estimators': [1, 10, 100, 250, 500, 1000]
    }


class TSF(Model):
    clf = TimeSeriesForestClassifier(verbose=0,
                                     n_jobs=-1,
                                     oob_score= True,
                                     bootstrap= True)
    clf_name = 'TSF'
    hyper_param = {'n_estimators': [1, 10, 100, 250, 500, 1000],
                   'max_features': ['sqrt', 'log2']
                   }


class LS(pytsModel):
    clf = LearningShapelets(verbose=0,
                            n_jobs=-1)
    clf_name = "LS"
    hyper_param = {
        'n_shapelets_per_size': [0.05, 0.15, 0.2, 0.3],
        'min_shapelet_length': [0.025, 0.075, 0.1, 0.125, 0.175, 0.2],
        'shapelet_scale': [1, 2, 3],
        'C': [5000, 2000, 1000, 100, 10, 1],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_iter':  [1000, 2000, 5000, 10000],
        'tol': [1e-3, 1e-2]
    }


class ST(Model):
    clf = ShapeletTransformClassifier(time_contract_in_mins=2)
    clf_name = "ST"
    hyper_param = {
        'time_contract_in_mins': [1, 2, 5],
        'n_estimators': [1, 100, 200, 500, 1000]
    }


class WEASEL(Model):
    clf = WEASEL(p_threshold=0.05,
                 n_jobs=-1)
    clf_name = "WEASEL"
    hyper_param = {
        # 'anova ': [True, False],
        'bigrams': [True, False],
        'binning_strategy': ['equi-depth', 'equi-width', 'information-gain'],
        'window_inc': [1, 2, 3, 4]
    }


class CBOSS(Model):
    clf = ContractableBOSS()
    clf_name = 'CBoss'
    hyper_param = {
        'n_parameter_samples': [100, 250, 500],
        'max_ensemble_size': [10, 25, 50]
    }


class INCEPTION(Model):
    clf = InceptionTimeClassifier(model_name='inception',
                                  verbose=False)
    clf_name = 'Inception'
    hyper_param = {
        'nb_filters': [32],
        'use_residual': [True],
        'use_bottleneck': [True],
        'bottleneck_size': [32],
        'depth': [6],
        'kernel_size': [41 - 1],
        'batch_size': [64],
        'nb_epochs': [1500]
    }

    def export_model(self, exp):
        cv = 'cv' if self.cv else 'no_cv'
        fname = os.path.join(self.models_folder,
                             f'{exp}_{self.clf_name}_{cv}.h5')
        self.clf.model.save(fname)
        self.logger.info(f'Model exported to {fname}')
