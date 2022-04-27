import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "/Users/ronkatz/Desktop/IML.HUJI/datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "/Users/ronkatz/Desktop/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(perceptron: Perceptron, x: np.ndarray, y_: int):
            losses.append(perceptron._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron._fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=range(len(losses)), y=losses, title=f"Loss when data is {n}")
        fig.update_traces(line_color='darksalmon')
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["/Users/ronkatz/Desktop/IML.HUJI/datasets/gaussian1.npy",
              "/Users/ronkatz/Desktop/IML.HUJI/datasets/gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        symbols = np.array(['x', 'diamond', 'circle'])

        from IMLearn.metrics import accuracy
        models = ["Gaussian Naive Bayes", "Linear Discriminant Analysis"]
        limits = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models])
        for i, m in enumerate([gnb, lda]):
            acc = round(accuracy(y, m._predict(X)), 3)
            fig.add_traces([decision_surface(m._predict, limits[0], limits[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(symbol=symbols[y.astype(int)], color=y,
                                                   colorscale=[custom[0], custom[-1], custom[2]]),
                                       text=f"accuracy: {acc}", textposition="middle center"),
                            go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers', marker=dict(color="black",
                                                                                                 symbol="x"),
                                       showlegend=False),
                            get_ellipse(m.mu_[0], m.cov_ if i == 1 else np.diag(m.vars_[0])),
                            get_ellipse(m.mu_[1], m.cov_ if i == 1 else np.diag(m.vars_[1])),  # add ellipses
                            get_ellipse(m.mu_[2], m.cov_ if i == 1 else np.diag(m.vars_[2]))],
                           rows=1, cols=i + 1)
            fig.layout.annotations[i].update(text=f"{models[i]} accuracy: {acc}")
        fig.update_layout(title=rf"$\textbf{{LDA and GNB estimators with gaussian dataset}}$",
                          margin=dict(t=100))
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
