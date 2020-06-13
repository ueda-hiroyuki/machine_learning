import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge

SAVE_PATH = "src/sample_data/AI_jobcolle/compare_regularization.png"


def main():
    X = np.linspace(-10, 10, 50)
    Y_gt = (X**3 + X**2 + X) * 0.001
    Y = Y_gt + np.random.normal(0, 0.05, len(X))

    poly = PolynomialFeatures(degree=30, include_bias=False)
    X_poly = poly.fit_transform(X[:, np.newaxis])

    xs = np.linspace(-10, 10, 200)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    t_poly = poly.fit_transform(xs[:, np.newaxis])

    model_plain = LinearRegression(normalize=True)
    model_plain.fit(X_poly, Y)
    y_plain = model_plain.predict(t_poly)

    print("#####################Plain#################")
    print(model_plain.coef_ * 10000)
    print('x{}\t{}'.format(0, model_plain.intercept_))
    for i, _c in enumerate(model_plain.coef_):
        print('x{}\t{}'.format(i+1, _c))
    print("###########################################")

    model_lasso = Lasso(normalize=True, alpha=0.01)
    model_lasso.fit(X_poly, Y)
    y_lasso = model_lasso.predict(t_poly)

    print("#####################Lasso#################")
    print(model_lasso.coef_ * 10000)
    print('x{}\t{}'.format(0, model_lasso.intercept_))
    for i, _c in enumerate(model_lasso.coef_):
        print('x{}\t{}'.format(i+1, _c))
    print("###########################################")

    model_ridge = Ridge(normalize=True, alpha=0.1)
    model_ridge.fit(X_poly, Y)
    y_ridge = model_ridge.predict(t_poly)

    print("#####################Ridge#################")
    print(model_ridge.coef_ * 10000)
    print('x{}\t{}'.format(0, model_ridge.intercept_))
    for i, _c in enumerate(model_ridge.coef_):
        print('x{}\t{}'.format(i+1, _c))
    print("###########################################")

    plt.figure(figsize=(10, 10))
    plt.plot(X, Y_gt, color='gray', label='ground truth')
    plt.plot(xs, y_plain, color='r', markersize=2, label='No Regularization')
    plt.plot(xs, y_lasso, color='g',  markersize=2, label='Lasso')
    plt.plot(xs, y_ridge, color='b',  markersize=2, label='Ridge')
    plt.plot(X, Y, '.', color='k')
    plt.legend()
    plt.ylim(-1, 1)

    plt.savefig(SAVE_PATH)

if __name__ == "__main__":
    main()


