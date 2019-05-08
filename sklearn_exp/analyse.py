#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     learning_curve, train_test_split)
def plot_learning_curve(estimator, X, y, title, ylim=None, cv='auto',
                        n_jobs=None, train_sizes='auto'):
    if cv=='auto':
        cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    if train_sizes=='auto':
        train_sizes = np.linspace(.05, 1.0, 10)
    
    # make plot area
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # calcurate lerning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #plt.grid()

    # plot learning curve
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    #plt.legend(loc="best")
    # legend at upper left outside of graph area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return plt

import copy
def importance_random(predictor, X, y, retry_num=10, random_state=None):
    feature_labels = X.columns
    original_score = predictor.score(X, y)
    sample_num = len(X)
    rnd = np.random
    importance_s = []
    std_s = []
    for feature in feature_labels: 
        diff_s_s = []
        tester = copy.copy(X)
        for i in range(retry_num):
            if random_state != None: rnd.seed(random_state + i)
            tester[feature] = rnd.uniform(X[feature].min(), X[feature].max(), sample_num)
            diff_s_s.append( original_score - predictor.score(tester, y) )
        importance_s.append(np.mean(diff_s_s))
        std_s.append(np.std(diff_s_s))
    return np.array(importance_s), np.array(std_s)

def figure_importances(feature_labels, feature_importances, importances_err=None, title=None):
    indices = np.argsort(feature_importances)
    plt.figure(figsize=(6,6))
    plt.barh(range(len(indices)), feature_importances[indices],
        xerr=None if importances_err is None else importances_err[indices], 
        capsize=3, color='cornflowerblue', align='center')
    plt.yticks(range(len(indices)), feature_labels[indices])
    if title is not None:
        plt.title(title)
    return plt

#%%
if __name__=="__main__":
    import tester
    x,y = tester.generate_suitably()

    from sklearn.kernel_ridge import KernelRidge
    rgr = KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)
    tester.predictor_parameter_test(predictor=rgr, params={'alpha':[0, 1.0, 10.0], 'gamma':[1.0]})
    plt.show()

    #cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    #t_size = np.linspace(.05, 1.0, 20)
    plt = plot_learning_curve(rgr, x, y, "title")
    plt.show()

#%%
if __name__=="__main__":
    import tester
    X,y = tester.generate_suitably3D()

    from sklearn.kernel_ridge import KernelRidge
    krr = KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)
    #tester.predictor_parameter_test(predictor=rgr, params={'alpha':[0, 1.0, 10.0], 'gamma':[1.0]})
    #plt.show()

    #cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    #t_size = np.linspace(.05, 1.0, 20)
    plot_learning_curve(krr, X, y, title="Kernel Ridge")
    plt.show()

    #train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.2)
    krr.fit(X, y)
    m,s = importance_random(krr, X, y)
    figure_importances(X.columns, m, s, title="KR")
    plt.show()

    from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              RandomForestRegressor)
    rfr = RandomForestRegressor(n_estimators=1000)
    #plot_learning_curve(rfr, X, y, title="Random Forest")
    #plt.show()

    rfr.fit(X,y)
    m,s = importance_random(rfr, X, y)
    figure_importances(X.columns, m, s, title="RF")
    plt.show()
    