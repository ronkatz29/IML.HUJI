import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "firefox"
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, header=0, parse_dates=True).dropna().drop_duplicates()
    X = X[X["Temp"] > -70]
    X = X[X["Day"] == pd.DatetimeIndex(X['Date']).day]
    X = X[X["Month"] == pd.DatetimeIndex(X['Date']).month]
    X = X[X["Year"] == pd.DatetimeIndex(X['Date']).year]
    X['DayOfYear'] = X.apply(lambda row: get_day(row.Month, row.Day), axis=1)
    return X

def get_day(month, day):
    return day + sum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:month])



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('/Users/ronkatz/Desktop/IML.HUJI/datasets/test_tmp.csv')
    # Question 2 - Exploring data for specific country
    israel = data[data["Country"] == "Israel"]
    fig = px.scatter(israel, x="DayOfYear", y="Temp", color="Year",
                     color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"],
                     title="Temperature as function of 'Day of year'")
    fig.show()

    monthly_std = israel.groupby('Month')['Temp'].std().rename("Standard deviation")
    fig = px.bar(monthly_std, title="Standard deviation of each month",)
    fig.show()

    # # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    #
    # # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    #
    # # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()