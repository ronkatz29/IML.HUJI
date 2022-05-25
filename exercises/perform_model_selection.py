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
    x = np.linspace(-1.2, 2, n_samples)
    polinom = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    poli_noise = polinom + np.random.normal(0, noise, n_samples)
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(poli_noise), 2 / 3)

    x_train, y_train, x_test, y_test = np.array(x_train).flatten(), np.array(y_train), np.array(x_test).flatten(), \
                                       np.array(y_test)

    go.Figure([go.Scatter(y=y_train, x=x_train.flatten(), mode='markers', name="$y_train$",
                          marker=dict(color='LightSkyBlue',size=10,line=dict(color='MediumPurple',width=2))),
               go.Scatter(y=y_test, x=x_test.flatten(), mode='markers', name="$y test$",
                          marker=dict(color='black',size=10,line=dict(color='red',width=2))),
               go.Scatter(y=polinom, x=x, mode='markers + lines', name="$y$")], layout=go.Layout(title=r"$\text{"
                                                                                                       r"data "
                                                                                                       r"visaluation}$",
                                                                                                 yaxis_title="y's",
                                                                                                 xaxis_title="x")).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err_all = []
    v_error_all = []
    for k in range(11):
        estimator = PolynomialFitting(k)
        train_err, v_error = cross_validate(estimator, x_train, y_train, mean_square_error, 5)
        train_err_all.append(train_err)
        v_error_all.append(v_error)

    go.Figure([go.Scatter(y=train_err_all, x=list(range(11)), mode='markers+lines', name="$train_error$"),
               go.Scatter(y=v_error_all, x=list(range(11)), mode='markers+lines', name="$validation_error$")],
              layout=go.Layout(title=r"$\text{"r"poli validation and train error for x samples}$", yaxis_title="err",
                               xaxis_title="k")).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = v_error_all.index(min(v_error_all))
    train_poly_model = PolynomialFitting(int(best_k)).fit(x_train, y_train)
    test_err = round(mean_square_error(y_test, train_poly_model.predict(x_test)), 2)
    print("val of k* is: " + str(best_k))
    print("test err: " + str(test_err))





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
    X,y = datasets.load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = X[:50], X[50:], y[:50], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamdot = np.linspace(0,2, n_evaluations)
    train_err_ridge = []
    v_error_ridge = []
    train_err_lasso = []
    v_error_lasso = []
    for lamda in lamdot:
        estimator = RidgeRegression(lamda)
        train_err, v_error = cross_validate(estimator, x_train, y_train, mean_square_error, 5)
        train_err_ridge.append(train_err)
        v_error_ridge.append(v_error)
        estimator = Lasso(lamda)
        train_err, v_error = cross_validate(estimator, x_train, y_train, mean_square_error, 5)
        train_err_lasso.append(train_err)
        v_error_lasso.append(v_error)

    go.Figure([go.Scatter(y=train_err_ridge, x=lamdot, mode='markers+lines', name="$train_error$"),
               go.Scatter(y=v_error_ridge, x=lamdot, mode='markers+lines', name="$validation_error$")],
              layout=go.Layout(title=r"$\text{"r"ridge validation and train error for x samples}$", yaxis_title="err",
                               xaxis_title="lamda")).show()

    go.Figure([go.Scatter(y=train_err_lasso, x=lamdot, mode='markers+lines', name="$train_error$"),
               go.Scatter(y=v_error_lasso, x=lamdot, mode='markers+lines', name="$validation_error$")],
              layout=go.Layout(title=r"$\text{"r"lasso validation and train error for x samples}$", yaxis_title="err",
                               xaxis_title="lamda")).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lamda_ridge = lamdot[np.argmin(np.array(v_error_ridge))]
    best_lamda_lasso = lamdot[np.argmin(np.array(v_error_lasso))]
    print("best lamda for ridge---> " + str(best_lamda_ridge))
    print("best lamda for lasso---> " + str(best_lamda_lasso))
    fit_ridge_model = RidgeRegression(best_lamda_ridge).fit(x_train, y_train)
    fit_lasso_model = Lasso(best_lamda_lasso).fit(x_train, y_train)
    fit_liner_model = LinearRegression().fit(x_train, y_train)
    err_ridge = mean_square_error(y_test, fit_ridge_model.predict(x_test))
    err_lasso = mean_square_error(y_test, fit_lasso_model.predict(x_test))
    err_liner = fit_liner_model.loss(x_test, y_test)
    print("Ridge err----> " + str(err_ridge))
    print("Lasso err----> " + str(err_lasso))
    print("Liner err----> " + str(err_liner))



if __name__ == '__main__':
    np.random.seed(0)
    # ## questions 1-5
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    ## questions 6-8
    select_regularization_parameter()

