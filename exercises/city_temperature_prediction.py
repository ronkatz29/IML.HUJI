import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "firefox"
pio.templates.default = "simple_white"

m_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


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
    X = X[X["Day"] > 0]
    X = X[X["Month"] > 0]
    X = X[X["Year"] > 0]
    X = X[X["Day"] < 32]
    X = X[X["Month"] < 13]
    X = X[X["Year"] < 2023]
    X = X[X["Day"] == pd.DatetimeIndex(X['Date']).day]
    X = X[X["Month"] == pd.DatetimeIndex(X['Date']).month]
    X = X[X["Year"] == pd.DatetimeIndex(X['Date']).year]
    X['DayOfYear'] = X.apply(lambda row: row.Day + sum(m_days[:row.Month]), axis=1)
    return X


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('/Users/ronkatz/Desktop/IML.HUJI/datasets/City_Temperature.csv')
    # Question 2 - Exploring data for specific country
    israel = data[data["Country"] == "Israel"]
    fig = px.scatter(israel, x="DayOfYear", y="Temp", color=israel['Year'].astype(str),
                     title="Temperature in relation with day of year")
    fig.show()

    monthly_std = israel.groupby('Month')['Temp'].std().rename('SD')
    fig = px.bar(monthly_std, title="SD of each month")
    fig.update_traces(marker_color='salmon')
    fig.show()

    # Question 3 - Exploring differences between countries
    month = data.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    month.columns = ['Country', 'Month', 'T_mean', 'T_std']
    fig = px.line(month, x='Month', y='T_mean', color=month['Country'].astype(str), title='Average temp for month per'
                                                                                          ' country with SD',
                  error_y='T_std')
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    israel = data[data["Country"] == "Israel"]
    israel_t = israel.Temp
    israel_d = israel.DayOfYear
    train_x, train_y, test_x, test_y = split_train_test(israel_d, israel_t)
    loss = []
    for k in range(1, 11):
        estimator = PolynomialFitting(k)
        estimator.fit(train_x.to_numpy().flatten(), train_y.to_numpy().flatten())
        loss.append(round(estimator.loss(train_x.to_numpy().flatten(), train_y.to_numpy().flatten()), 2))

    for ind, val in enumerate(loss):
        print("test error recorded for k=" + str(ind + 1) + " is " + str(val))

    fig = px.bar(x=range(1, 11), y=loss, title="Loss for k in range 10",
                 labels={"x": "dgree", "y": "MSE"})
    fig.update_traces(marker_color='purple')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    p_loss = []
    estimator = PolynomialFitting(5)
    estimator.fit(israel_d.to_numpy(), israel_t.to_numpy())

    data_i = data[data['Country'] == 'Jordan']
    loss = estimator.loss(data_i.DayOfYear.to_numpy(), data_i.Temp.to_numpy())
    p_loss.append(loss)

    data_n = data[data['Country'] == 'The Netherlands']
    loss = estimator.loss(data_n.DayOfYear.to_numpy(), data_n.Temp.to_numpy())
    p_loss.append(loss)

    data_sa = data[data['Country'] == 'South Africa']
    loss = estimator.loss(data_sa.DayOfYear.to_numpy(), data_sa.Temp.to_numpy())
    p_loss.append(loss)

    fig = px.bar(x=['Jordan', 'The Netherlands', 'South Africa'], y=p_loss, title="Israel error over else countries",
                 labels={"x": "country", "y": "MSE val of israel"})
    fig.update_traces(marker_color='green')
    fig.show()
