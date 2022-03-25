from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "firefox"
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_gaussian = UnivariateGaussian()
    mu, sigma, sep_num = 10, 1 , 1000
    sample = np.random.normal(mu, sigma, sep_num)
    uni_gaussian.fit(sample)
    print(uni_gaussian.mu_, uni_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = []
    for i in range(10, 1001, 10):
        ms.append(i)
    estimated_mean = []
    for m in ms:
        estimated_mean.append(abs(np.mean(sample[:m]) - 10))

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Estimation of Expectation As Function Of "
                        r"Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="distance est' and true",
                  height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_arr = uni_gaussian.pdf(sample)
    go.Figure([go.Scatter(x=sample, y=pdf_arr, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{PDF Scatter Graph}$",
                  xaxis_title="pdf",
                  yaxis_title="normal distribution sample",
                  height=600)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mult_gaussian = MultivariateGaussian()
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    mu = np.array([0, 0, 4, 0]).transpose()
    s = np.random.multivariate_normal(mu, sigma, 1000)
    mult_gaussian.fit(s)
    print(mult_gaussian.mu_)
    print(mult_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f_1 = np.linspace(-10, 10, 200)
    f_3 = np.linspace(-10, 10, 200)
    log_lh_mat = np.array([[mult_gaussian.log_likelihood
                              (np.array([x, 0, y, 0]), sigma, s)
                              for x in f_1] for y in f_3])

    go.Figure(data=go.Heatmap(x=f_3, y=f_1, z=log_lh_mat,
                              colorscale='rainbow'),
              layout=go.Layout(title='Heatmap of log-likelihood of drawn '
                                     'samples ',
              xaxis_title="f1_scale(vals)",
              yaxis_title='f3_scale(vals) ')).show()

    # Question 6 - Maximum likelihood
    max_vals = np.unravel_index(np.argmax(log_lh_mat),
                                np.shape(log_lh_mat))
    max_val = log_lh_mat[139][99]
    f1_max = f_1[max_vals[0]]
    f3_max = f_3[max_vals[1]]
    # print(max_val, f1_max, f3_max)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
