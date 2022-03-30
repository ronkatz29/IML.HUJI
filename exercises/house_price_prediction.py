from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "firefox"
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename, header=0, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).dropna()

    for feature in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built"]:
        df = df[df[feature] > 0]
    for feature in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[feature] >= 0]

    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """

    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    def pearson_corr(x,y):
        cov = np.cov(x, y)[0, 1]
        var_x, var_y = np.std(x), np.std(y)
        return cov / (var_x * var_y)

    for feature in X:
        corr = pearson_corr(X[feature],y)
        plot =  px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y",
                         title=f"Correlation Between {feature} val and Response <br> Corr {corr}",
                         labels={"x": f"{feature} Values", "y": "Response Values"})
        pio.write_image(plot, output_path + "pearson.correlation.%s.png" % feature)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("/Users/ronkatz/Desktop/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df.drop('price', 1), df.price, "")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(df.drop('price', 1), df.price, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # fitted_data = []
    # for i in range(10,101):
    #     for p in range(10):



