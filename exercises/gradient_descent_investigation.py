import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

COLOR = ["red", "green", "blue", "black"]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def call_back(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return call_back, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    lowest_gd1, lowest_gd2 = 2, 2
    best_eta_l1 = 0
    best_eta_l2 = 0

    for eta in etas:
        l1 = L1(init.copy())
        l2 = L2(init.copy())
        c_v_w1 = get_gd_state_recorder_callback()
        c_v_w2 = get_gd_state_recorder_callback()

        gr1 = GradientDescent(learning_rate=FixedLR(eta), callback=c_v_w1[0], out_type="best")
        gr1.fit(l1, X=None, y=None)
        gr2 = GradientDescent(learning_rate=FixedLR(eta), callback=c_v_w2[0], out_type="best")
        gr2.fit(l2, X=None, y=None)

        fig1 = plot_descent_path(L1, np.array(c_v_w1[2]), title="L1 with eta " + str(eta))
        fig1.show()
        fig2 = plot_descent_path(L2, np.array(c_v_w2[2]), title="L2 with eta " + str(eta))
        fig2.show()

        fig3 = go.Figure([go.Scatter(x=list(range(len(c_v_w1[1]))), y=c_v_w1[1], mode="markers", marker_color="black")],
                         layout=go.Layout(title="convergence rate of L1 with eta " + str(eta),
                                          xaxis_title="iter num", yaxis_title="convergence val"))
        fig3.show()
        fig4 = go.Figure([go.Scatter(x=list(range(len(c_v_w2[1]))), y=c_v_w2[1], mode="markers", marker_color="red")],
                         layout=go.Layout(title="convergence rate of L2 with eta " + str(eta),
                                          xaxis_title="iter num", yaxis_title="convergence val"))
        fig4.show()

        if (c_v_w1[1][-1] < lowest_gd1):
            lowest_gd1 = c_v_w1[1][-1]
            best_eta_l1 = eta

        if (c_v_w2[1][-1] < lowest_gd2):
            lowest_gd2 = c_v_w2[1][-1]
            best_eta_l2 = eta

    print("bestloss l1 is --->" + str(lowest_gd1) + " with eta " + str(best_eta_l1))
    print("bestloss l2 is ---> " + str(lowest_gd2) + " with eta " + str(best_eta_l2))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    best_gam = 0
    lowest_v = 2
    all_rate = []
    w_g95 = []
    for gam in gammas:
        l1 = L1(init.copy())
        c_v_w1 = get_gd_state_recorder_callback()

        gr1 = GradientDescent(learning_rate=ExponentialLR(eta, gam), callback=c_v_w1[0], out_type="best")
        gr1.fit(l1, X=None, y=None)
        all_rate.append(c_v_w1[1])
        if gam == .95:
            w_g95 = c_v_w1[2]
        if (c_v_w1[1][-1] < lowest_v):  # check if last value in values is a better loss for l1
            lowest_v = c_v_w1[1][-1]
            best_gam = gam

    go.Figure(
        [go.Scatter(x=list(range(len(all_rate[0]))), y=all_rate[0], mode="markers+lines", marker_color=COLOR[0],
                    name="gam-->" + str(gammas[0])),
         go.Scatter(x=list(range(len(all_rate[1]))), y=all_rate[1], mode="markers+lines", marker_color=COLOR[1],
                    name="gam-->" + str(gammas[1])),
         go.Scatter(x=list(range(len(all_rate[2]))), y=all_rate[2], mode="markers+lines", marker_color=COLOR[2],
                    name="gam-->" + str(gammas[2])),
         go.Scatter(x=list(range(len(all_rate[3]))), y=all_rate[3], mode="markers+lines", marker_color=COLOR[3],
                    name="gam-->" + str(gammas[3]))],
        layout=go.Layout(title="CONVERGENCE RATES FOR DIFF GAMMA VALUES",
                         xaxis_title="iter num", yaxis_title="convergence val")).show()

    print("bestloss is --->" + str(lowest_v) + " with gma " + str(best_gam))

    # Plot descent path for gamma=0.95
    l2 = L2(init.copy())
    c_v_w2 = get_gd_state_recorder_callback()
    gr2 = GradientDescent(learning_rate=ExponentialLR(eta, .95), callback=c_v_w2[0], out_type="best")
    gr2.fit(l2, X=None, y=None)

    plot_descent_path(L1, np.array(w_g95), title="gd path l1 model with gam 0.95 ").show()
    plot_descent_path(L2, np.array(c_v_w2[2]), title="gd path l2 model with gam 0.95 ").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    from IMLearn.metrics import misclassification_error
    from IMLearn.model_selection import cross_validate
    from sklearn.metrics import roc_curve, auc
    from utils import custom

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    lg = LogisticRegression(include_intercept=True, solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    lg._fit(X_train, y_train)
    label_prob = lg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, label_prob)
    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_a = round(thresholds[np.argmax(tpr - fpr)], 2)
    print("best alpha is---->" + str(best_a))
    lg.alpha_ = best_a
    lg_test_error = lg._loss(X_test, y_test)
    print("models test error--->" + str(lg_test_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lamdas = [0.1, 0.002, 0.02, 0.001, 0.005, 0.01, 0.05]
    v_error_all_1 = []
    v_error_all_2 = []
    for lamda in lamdas:
        print("starting " + str(lamda))
        estimator1 = LogisticRegression(include_intercept=True, penalty="l1",
                                       solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), lam=lamda)
        train_err2, v_error1 = cross_validate(estimator1, X_train, y_train, misclassification_error)
        v_error_all_1.append(v_error1)
        estimator2 = LogisticRegression(include_intercept=True, penalty="l2",
                                        solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), lam=lamda)
        train_err2, v_error2 = cross_validate(estimator2, X_train, y_train, misclassification_error)
        v_error_all_2.append(v_error2)

    best_lam_1 = lamdas[np.argmin(v_error_all_1)]
    lr_l1 = LogisticRegression(include_intercept=True, penalty="l1",
                               solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                               lam=best_lam_1)
    print()
    print("best lamda l1 is---->" + str(best_lam_1))
    lr_test_error = lr_l1.fit(X_train, y_train)._loss(X_test, y_test)
    print("l1 model test error--->" + str(lr_test_error))
    print()
    best_lam_2 = lamdas[np.argmin(v_error_all_2)]
    lr_l2 = LogisticRegression(include_intercept=True, penalty="l2",
                               solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), lam=best_lam_2)

    print()
    print("best lamda l2 is---->" + str(best_lam_2))
    lr_test_error = lr_l2.fit(X_train, y_train)._loss(X_test, y_test)
    print("l2 model test error--->" + str(lr_test_error))



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    # fit_logistic_regression()

