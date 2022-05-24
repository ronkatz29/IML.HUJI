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
    #split data
    sample_split = np.array_split(X, cv)
    all_errors = np.zeros(cv)
    val_errors = np.zeros(cv)

    for i in range(cv):
        train_x = np.concatenate(sample_split[:i] + sample_split[i+1:], axis=0)
        train_v = sample_split[i]
        all_errors[i] = scoring(estimator.fit(train_x, y[train_x]).predict(train_x), y[train_x])
        val_errors[i] = scoring(estimator.fit(train_v, y[train_v]).predict(train_v), y[train_v])

    return all_errors.mean(), val_errors.mean()






