from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sktime.classification.distance_based._time_series_neighbors import KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based._elastic_ensemble import ElasticEnsemble
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.interval_based import TimeSeriesForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import BOSSIndividual

from sklearn.metrics import balanced_accuracy_score

class MyAutoMLClassifier:
  def __init__(self, scoring_function = 'balanced_accuracy', n_iter = 50):
    self.scoring_function = scoring_function
    self.n_iter = n_iter
  
  def fit(self,X,y):
    X_train = X
    y_train = y

    model_pipeline_steps = []
    model_pipeline_steps.append(('estimator',LogisticRegression()))
    model_pipeline = Pipeline(model_pipeline_steps)

    optimization_grid = []

    # 1NN-MSM
    optimization_grid.append({
        'estimator':[KNeighborsTimeSeriesClassifier(metric='msm')],
        'estimator__n_neighbors': [1],
        'estimator__weights': ['uniform','distace'],
        'estimator__algorithm':['brute']
    })

    search = RandomizedSearchCV(
      model_pipeline,
      optimization_grid,
      n_iter=self.n_iter,
      scoring = self.scoring_function, 
      n_jobs = -1, 
      random_state = 0, 
      verbose = 3,
      cv = 5
    )

    import IPython
    IPython.embed()

    search.fit(X_train, y_train)
    self.best_estimator_ = search.best_estimator_
    self.best_pipeline = search.best_params_

  def predict(self,X,y = None):
    return self.best_estimator_.predict(X)

  def predict_proba(self,X,y = None):
    return self.best_estimator_.predict_proba(X)