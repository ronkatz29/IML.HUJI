from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    epsilon = np.random.normal(0, noise, n_samples)
    x = np.linspace(-1.2, 2, n_samples)
    polinom = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    poli_noise = polinom + np.random.normal(0, noise, n_samples)
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(poli_noise), 2 / 3)

    go.Figure([go.Scatter(y=polinom, x=x, mode='markers + lines', name="$Train loss$"),
               go.Scatter(y=poli_noise, x=x, mode='markers + lines', name="$Validation loss$")],
              layout=go.Layout(title=r"$\text{Train losses}$",
                               height=650, yaxis_title="Loss on train",
                               xaxis_title="Polynomial degree")).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
