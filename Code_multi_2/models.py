from time import process_time
import os
import logging
import pandas as pd
import joblib
import numpy as np

# from scipy.sparse.construct import rand
# from scipy.stats.stats import _euclidean_dist
from scipy.spatial.distance import euclidean
from scipy.stats.stats import mode
# from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sktime_dl.deeplearning.base import estimators
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
from sktime.transformations.panel.shapelets import ContractedShapeletTransform

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
            n_jobs=-2,
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
        models_folder = os.path.join(os.getcwd(), 'datasets', ds, 'models','')
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.models_folder = models_folder
        if self.dim > 1:
            mv_estimators, mv_param_dict = self.get_params_dict()
            self.clf = ColumnEnsembleClassifier(estimators=mv_estimators, remainder="drop", verbose=False)
            self.hyper_param = mv_param_dict
        else:
            pass

    def get_params_dict(self):
        estimators = []
        params_dict = {}
        for i in range(self.dim):
            temp = f"{self.clf_name}_{i}__"

            estimators.append((
                f"{self.clf_name}_{i}",
                self.clf,
                [i]))

            for key, val in self.hyper_param.items():
                params_dict[temp + str(key)] = val
        return estimators, params_dict

    def export_model(self, process, revealed_pct):
        cv = 'cv' if self.cv else 'no_cv'
        pct = f"{int(revealed_pct)}_pct" if revealed_pct else None
        fname = os.path.join(self.models_folder,
                             f'{process}_{self.clf_name}_{cv}_{pct}.sav' if pct
                             else f'{process}_{self.clf_name}_{cv}.sav')
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
        models_folder = os.path.join(os.getcwd(), 'datasets', ds, 'models','')
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
                               weights='uniform',
                               algorithm='auto',
                               leaf_size=30,
                               p=2,
                               metric='minkowski',
                               n_jobs=-2)
    clf_name = "1NN-ED"
    hyper_param = {
        'n_neighbors' : [1],
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'brute'],
        'leaf_size' : [1, 10, 30, 100],
        'p' : [2],
        'metric' : ['minkowski'],
        'n_jobs' : [-2]
    }


class KNNED_SKTIME(Model):
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1,
                                         metric=euclidean)
    clf_name = "1NN-ED-SKTime"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance']
    }

class KNNDTW(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               weights='uniform',
                               algorithm='auto',
                               leaf_size=30,
                               p=2,
                               metric='dtw',
                               n_jobs=-1)
    clf_name = "1NN-DTW"
    hyper_param = {
        'n_neighbors' : [1],
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'brute'],
        'leaf_size' : [1, 10, 30, 100],
        'p' : [2],
        'metric' : ['dtw'],
        'n_jobs' : [-1]
    }

class KNNDTW_sc(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='dtw_sakoechiba',
                               n_jobs=-2)
    clf_name = "1NN-DTW-sakoechiba"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        'n_jobs' : [-2]
    }

class KNNDTW_it(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='dtw_itakura',
                               n_jobs=-2)
    clf_name = "1NN-DTW-itakura"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        'n_jobs' : [-2]
    }

class KNNDTW_ms(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='dtw_multiscale',
                               n_jobs=-2)
    clf_name = "1NN-DTW-multiscale"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        'n_jobs' : [-2]
    }

class KNNDTW_fs(pytsModel):
    clf = KNeighborsClassifier(n_neighbors=1,
                               metric='dtw_fast',
                               n_jobs=-2)
    clf_name = "1NN-DTW-fast"
    hyper_param = {
        'algorithm': ['auto', 'brute'],
        'weights': ['uniform', 'distance'],
        'n_jobs' : [-2]
    }

class KNNMSM(Model):
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=1,
                                         weights="uniform",
                                         algorithm="brute",
                                         metric="msm")
    clf_name = "1NN-MSM"
    hyper_param = {
        'n_neighbors' : [1],
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'brute'],
        # 'metric' : ['msm'],
        'metric_params': [{'c': 0.01}, {'c': 0.1}, {'c': 1}, {'c': 10}, {'c': 100}]
    }

class EE(Model):
    clf = ElasticEnsemble(verbose=0)
    clf_name = 'EE'
    hyper_param = {
        'distance_measures': ['all']
    }


class PFOREST(Model):
    clf = ProximityForest(random_state=None,
                        n_estimators=100,
                        distance_measure=None,
                        get_distance_measure=None,
                        verbosity=0,
                        max_depth=np.math.inf,
                        n_jobs=16,
                        n_stump_evaluations=5,
                        find_stump=None)
    clf_name = 'PForest'
    hyper_param = {
        'random_state' : [None],
        'n_estimators' : [100],
        'distance_measure' : [None],
        'get_distance_measure' : [None],
        'verbosity' : [0],
        'max_depth' : [np.math.inf],
        'n_jobs' : [16],
        'n_stump_evaluations' : [5],
        'find_stump' : [None]
        }


class TSF(Model):
    clf = TimeSeriesForestClassifier(estimator=None,
                                    n_estimators=500,
                                    criterion='entropy',
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0,
                                    max_features='sqrt',
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    min_impurity_split=None,
                                    bootstrap=True,
                                    oob_score=True,
                                    n_jobs=-2,
                                    random_state=None,
                                    verbose=0,
                                    warm_start=False,
                                    class_weight=None,
                                    max_samples=None)
    clf_name = 'TSF'
    hyper_param = {
                    'estimator' : [None],
                    'n_estimators' : [500],
                    'criterion' : ['entropy', 'gini'],
                    'max_depth' : [None],
                    'min_samples_split' : [2],
                    'min_samples_leaf' : [1],
                    'min_weight_fraction_leaf' : [0],
                    'max_features': ['sqrt', 'log2'],
                    'max_leaf_nodes' : [None],
                    'min_impurity_decrease' : [0],
                    'min_impurity_split' : [None],
                    'bootstrap' : [True],
                    'oob_score' : [True],
                    'n_jobs' : [-2],
                    'random_state' : [None],
                    'verbose' : [0],
                    'warm_start' : [False],
                    'class_weight' : [None],
                    'max_samples' : [None]
                   }


class LS(pytsModel):
    clf = LearningShapelets(min_shapelet_length=0.1,
                            shapelet_scale=3,
                            penalty='l2',
                            tol=0.001,
                            C=1000,
                            learning_rate=0.01,
                            max_iter=10000,
                            multi_class='ovr',
                            alpha=- 100,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=None,
                            verbose=0,
                            random_state=None,
                            n_jobs=-2)
                            
    clf_name = "LS"
    hyper_param = {
        'n_shapelets_per_size' : [0.05, 0.15, 0.3],
        'min_shapelet_length' : [0.025, 0.075, 0.1, 0.125, 0.175, 0.2],
        'shapelet_scale' : [1,2,3],
        'penalty' : ['l2'],
        'tol' : [0.001],
        'C' : [1000, 100, 10, 1],
        'learning_rate' : [0.01],
        'max_iter' : [2000, 5000, 10000],
        'multi_class' : ['ovr'],
        'alpha' : [- 100],
        'fit_intercept' : [True],
        'intercept_scaling' : [1],
        'class_weight' : [None],
        'verbose' : [0],
        'random_state' : [None],
        'n_jobs' : [-2]
    }


class ST(Model):
    clf = ShapeletTransformClassifier(time_contract_in_mins= 60)
    clf_name = "ST"
    hyper_param = {
        'n_estimators': [500, 1000]
    }

class ST_ensemble(Model):
    st= ContractedShapeletTransform(min_shapelet_length=3,
                                    max_shapelet_length=np.inf,
                                    max_shapelets_to_store_per_class=200,
                                    time_contract_in_mins=60,
                                    num_candidates_to_sample_per_case=20,
                                    random_state=None,
                                    verbose=0,
                                    remove_self_similar=True)
    nb = GaussianNB()
    tree= tree.DecisionTreeClassifier(max_features='sqrt')
    rf= RandomForestClassifier(n_estimators=500)
    svm_lin= SVC(kernel='linear')
    svm_rbf = SVC(kernel='rbf')
    vote= VotingClassifier(estimators=[
                                ('nb',nb),
                                ('tree',tree),
                                ('rf',rf),
                                ('svm_lin',svm_lin),
                                ('svm_rbf',svm_rbf)],
                              voting='hard',
                              weights=None,
                              n_jobs=-2,
                              flatten_transform=True,
                              verbose=False)
    steps = [('st',st), ("clf", vote)]

    clf = Pipeline(steps= steps)
    clf_name = "ST_ensemble"
    hyper_param = {}


class WEASEL(Model):
    clf = WEASEL(anova=True,
    bigrams=True,
    binning_strategy="information-gain",
    window_inc=2,
    p_threshold=0.05,
    n_jobs=-2,
    random_state=None)
    clf_name = "WEASEL"
    hyper_param = {
        'anova':[True, False],
        'bigrams':[True, False],
        'binning_strategy': ['equi-depth', 'equi-width', 'information-gain'],
        'window_inc': [2,3,4],
        'n_jobs' : [-2],
        'p_threshold' : [0.05]
    }


class CBOSS(Model):
    clf = ContractableBOSS(n_parameter_samples=250,
                          max_ensemble_size=50,
                          max_win_len_prop=1,
                          time_limit=60,
                          min_window=5,
                          random_state=None)
    clf_name = 'CBoss'
    hyper_param = {
        'n_parameter_samples' : [250],
        'max_ensemble_size': [100, 250, 500],
        'max_win_len_prop' : [0.5, 1],
        'time_limit' : [60],
        'min_window' : [5],
        'random_state' : [None]
    }


class INCEPTION(Model):
    clf = InceptionTimeClassifier(nb_filters=32,
                                  use_residual=True,
                                  use_bottleneck=True,
                                  bottleneck_size=32,
                                  depth=6,
                                  kernel_size=41 - 1,
                                  batch_size=64,
                                  nb_epochs=1500,
                                  callbacks=None,
                                  random_state=0,
                                  verbose=False,
                                  model_name='inception',
                                  model_save_directory=None)
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

    def export_model(self, process, revealed_pct):
        cv = 'cv' if self.cv else 'no_cv'
        pct = f"{int(revealed_pct)}_pct" if revealed_pct else None
        fname = os.path.join(self.models_folder,
                             f'{process}_{self.clf_name}_{cv}_{pct}.h5' if pct
                             else f'{process}_{self.clf_name}_{cv}.h5')
        self.clf.model.save(fname)
        self.logger.info(f'Model exported to {fname}')
