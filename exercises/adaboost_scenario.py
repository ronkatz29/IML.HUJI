import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy
import matplotlib.pyplot as plt

pio.renderers.default = "firefox"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, 250)
    adaboost.fit(train_X, train_y)

    axis_x = np.linspace(1, 250, 250).astype(int)
    train_err = [adaboost.partial_loss(train_X, train_y, s) for s in axis_x]
    test_err = [adaboost.partial_loss(test_X, test_y, s) for s in axis_x]

    plt.plot(axis_x, train_err, label='train err')
    plt.plot(axis_x, test_err, label="test err")
    plt.title("AdaBoost err as func of num of fitted models")
    plt.xlabel("num of models")
    plt.ylabel("overall loss")
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    figure = make_subplots(2, 2, subplot_titles=[f"AdaBoost train -> {t} iterations" for t in T])
    for index, t in enumerate(T):
        traces = [decision_surface(lambda v: adaboost.partial_predict(v, t), lims[0], lims[1], showscale=False),
                  go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y),
                             showlegend=False)]
        figure.add_traces(traces, rows=int(index / 2) + 1, cols=(index % 2) + 1)
    figure.update_layout(title="AdaBoost performance with diff num of iterations\n")
    figure.show()

    # Question 3: Decision surface of best performing ensemble
    accuracies = np.array([accuracy(test_y, adaboost.partial_predict(test_X, T=t)) for t in T])
    arg_m = np.argmax(accuracies)

    xrange, yrange = np.linspace(*lims[0], 120), np.linspace(*lims[1], 120)
    xx, yy = np.meshgrid(xrange, yrange)
    prediction = adaboost.partial_predict(np.c_[xx.ravel(), yy.ravel()], T=T[arg_m]).reshape(
        xx.shape)
    y_colors = pd.DataFrame(test_y).replace({1: 'b', -1: 'g'}).to_numpy().reshape(len(test_y))
    plt.pcolormesh(xx, yy, prediction)
    plt.scatter(x=test_X[:, 0][y_colors == 'b'], y=test_X[:, 1][y_colors == 'b'],
                c=y_colors[y_colors == 'b'])
    plt.scatter(x=test_X[:, 0][y_colors == 'g'], y=test_X[:, 1][y_colors == 'g'],
                c=y_colors[y_colors == 'g'])
    plt.title(f"Decision Surface of the ensemble\n"
              f"size : {T[arg_m]} accuracy :" + str(accuracies[arg_m]))
    plt.show()

    # Question 4: Decision surface with weighted samples
    size_factor = 5
    if noise == 0:
        size_factor = 50
    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=(train_y == 1).astype(int),
                                                  size=size_factor * adaboost.D_ / np.max(adaboost.D_),
                                                  symbol=class_symbols[train_y.astype(int)],
                                                  colorscale=["black", "blue"],
                                                  line=dict(color="black", width=1)))
    fig = go.Figure([decision_surface(adaboost.predict, lims[0], lims[1],
                                      showscale=False, colorscale=["black", "blue"]), scatter])
    fig.update_xaxes(range=[-1, 1], constrain="domain")
    fig.update_yaxes(range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(dict1=dict(width=500, height=500, title="Decision Surface of best ensemble"))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
