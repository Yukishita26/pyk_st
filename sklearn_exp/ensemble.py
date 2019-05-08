import numpy as np
import pandas as pd
import matplotlib as plt
import sys

class WrongSizeError(Exception):
    def __init__(self, text="Size not match"):
        self.text = text

from typing import (List, Union, Optional)
Number = Union[int, float]
Ndarray = np.ndarray
ArrayLike = Union[List[List[Number]], Ndarray]

# hight x width の array_like を length x 1 の array に縮約
def mean_with_weight(arr:ArrayLike, weight:Optional[List[Number]] = None, axis:int = 0) -> Ndarray:
    if type(arr) == list: arr = np.array(arr)
    hight, width = arr.shape
    #length = arr.shape[1 - axis]
    vertical = arr.shape[axis]
    
    if weight == None:
        weight = np.ones(vertical)
    else:
        if len(weight) != vertical:
            raise WrongSizeError("weight doesn't match for array size")

    whole_weight = sum(weight)
    if axis == 0:
        return sum([arr[i,:] * weight[i] for i in range(hight)])/whole_weight
    else:
        return sum([arr[:,j] * weight[j] for j in range(width)])/whole_weight

if __name__ == "__main__":
    array = np.array([[1, 2, 3],
                      [4, 5, 6]])
    print(mean_with_weight(array, weight=[1, 0], axis=0))
    # => [1, 2, 3]

    print(mean_with_weight(array, weight=[1, 0, 1], axis=1))
    # => [2, 5]

    try:
        print(mean_with_weight(array, weight=[0, 1], axis=1))
    except WrongSizeError as wse:
        print(wse.text)
    # => weight doesn't match for array size


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import random
from math import ceil

class ExtendedForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 b_estimator='d',# decision or extra
                 boosting=False,#if True, AdaBoost, o.w. Bagging
                 random_state=None,
                 bootstrap=True,
                 bootstrap_features=False,
                 max_samples=1.0,
                 n_estimators=100,
                 ):
        if b_estimator=='d' or b_estimator=='decision':
            self.b_estimator = 'decision'
            self.base_estimator = DecisionTreeRegressor()
        elif b_estimator=='e' or b_estimator=='extra':
            self.b_estimator = 'extra'
            self.base_estimator = ExtraTreeRegressor()
        else:
            self.b_estimator = b_estimator
            self.base_estimator = b_estimator
        self.boosting = boosting
        self.random_state = random_state
        if random_state==None: self.random_state = random.randint(0,1000)
        else: self.random_state = random_state
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        
        self.sample_rate = self.max_samples/ceil(self.max_samples)
        if boosting:
            self.estimator_ = AdaBoostRegressor(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                #bootstrap=self.bootstrap,
                #bootstrap_features=self.bootstrap_features,
                #max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        else:
            self.estimator_ = BaggingRegressor(
                base_estimator=self.base_estimator,
                random_state=self.random_state,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                max_samples=self.sample_rate,
                n_estimators=self.n_estimators)
        #self.estimators_ = None
        #self.feature_importances_ = None

    def fit(self, X, y):
        if self.max_samples > 1:
            X = pd.concat([X]*ceil(self.max_samples))
            y = pd.concat([y]*ceil(self.max_samples))
        self.estimator_.fit(X, y)
        self.estimators_ = self.estimator_.estimators_
        if self.boosting:
            self.estimator_weights_ = self.estimator_.estimator_weights_
            self.estimator_errors_ = self.estimator_.estimator_errors_
            self.feature_importances_s_ = \
                np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = mean_with_weight(self.feature_importances_s_, weight=self.estimator_weights_)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        else:
            self.feature_importances_s_ = np.array([e.feature_importances_ for e in self.estimators_])
            self.feature_importances_ = self.feature_importances_s_.mean(axis=0)
            self.feature_importances_std_ = self.feature_importances_s_.std(axis=0)
        return self
    def predict(self, X):
        return self.estimator_.predict(X)
    def score(self, X, y):
        yhat = self.predict(X)
        return r2_score(y, yhat)