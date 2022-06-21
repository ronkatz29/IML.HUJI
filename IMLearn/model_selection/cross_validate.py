from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # split data
    # all_errors, val_errors = 0, 0
    # fold_s = int(y.shape[0]/cv)
    #
    # for i in range(cv):
    #     train_x = np.concatenate((X[:fold_s*i], X[fold_s*(i+1):]), axis=0)
    #     train_y = np.concatenate((y[:fold_s*i], y[fold_s*(i+1):]), axis=0)
    #     train_v = X[fold_s*i:fold_s*(i+1)]
    #     train_v_y = y[fold_s*i:fold_s*(i+1)]
    #     estimator.fit(train_x, train_y)
    #     # fited_v = estimator.fit(train_v, train_v_y)
    #     all_errors += scoring(estimator.predict(train_x), train_y)
    #     val_errors += scoring(estimator.predict(train_v), train_v_y)
    #
    # return all_errors/cv, val_errors/cv

    #scholl solution cross validation
    ids = np.arange(X.shape[0])
    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)
    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv








