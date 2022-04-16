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
    # choose col for model and drop some cols
    df = pd.read_csv(filename, header=0, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
                                                  ]).dropna()

    # remove not logical data
    df = df.drop_duplicates()
    for feature in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "bathrooms"]:
        df = df[df[feature] > 0]
    for feature in ["floors", "sqft_basement", "yr_renovated"]:
        df = df[df[feature] >= 0]

    # check data in right range
    df = df[df["waterfront"].isin([0, 1])]
    df = df[df["view"].isin(range(5))]
    df = df[df["condition"].isin(range(1, 6))]
    df = df[df["grade"].isin(range(1, 15))]
    df = df[df["bedrooms"] < 10]

    # get dummy values for zipcode
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

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
    for x in X:
        if "zipcode" in x:
            X = X.drop(columns=str(x))

    def pearson_corr(x, y):
        cov = np.cov(x, y)[0, 1]
        var_x, var_y = np.std(x), np.std(y)
        return cov / (var_x * var_y)

    # generate all the scatter for each feacher and culc each PC
    for feature in X:
        corr = pearson_corr(X[feature], y)
        plot = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y",
                          title="Correlation Between " + str(feature) + " val and Response Corr : " + str(corr),
                          labels={"x": feature, "y": "Response Val"})
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
    estimator = LinearRegression()
    average = []
    var = []
    for p in range(10, 101, 1):  # percentage p in 10%, 11%, ..., 100%,
        p_loss = []
        for nana in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # repeat the following 10 times
            p_train_x = train_x.sample(frac=p / 100)
            p_train_y = train_y.reindex_like(p_train_x)
            estimator.fit(p_train_x.to_numpy(), p_train_y.to_numpy())
            p_loss.append(estimator.loss(test_x.to_numpy(), test_y.to_numpy()))
        average.append(np.mean(p_loss))
        var.append(np.std(p_loss))
    x_axis = range(10, 101)
    var = np.array(var)
    average = np.array(average)

    frames = []
    for _ in range(10):
        frames.append(go.Frame(data=go.Scatter(x=list(x_axis), y=average, mode="markers+lines",
                                               name="Means",
                                               marker=dict(color="fuchsia", opacity=.7))))

    for i in range(len(frames)):
        frames[i]["data"] = (go.Scatter(x=list(x_axis), y=average - 2 * var, fill=None, mode="lines",
                                        line=dict(color="darkkhaki"), showlegend=False),
                             go.Scatter(x=list(x_axis), y=average + 2 * var, fill='tonexty', mode="lines",
                                        line=dict(color="darkkhaki"), showlegend=False),) + \
                            frames[i]["data"]
    fig = go.Figure(data=frames[0]["data"], frames=frames,
                    layout=go.Layout(title="MSE as a function of p%",
                                     xaxis=frames[0]["layout"]["xaxis"],
                                     yaxis=frames[0]["layout"]["yaxis"],
                                     xaxis_title="% of samples used",
                                     yaxis_title="average mean square error over 10 times"))
    fig.show()
