# 1. Use a linear regression model with the Boston housing data set. Your code should then return which factor
# has the largest effect on the price of housing in Boston. (This is not the correlation coefficient.
# This is the absolute value of the slope.)

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np


def createBostonModel(X, y):
    return LinearRegression().fit(X, y)


def whichFactor(reg, names):
    index = np.argmax(np.absolute(reg.coef_))
    return names[index]


if __name__ == "__main__":
    names = load_boston().feature_names
    X, y = load_boston(return_X_y=True)
    reg = createBostonModel(X, y)
    print(whichFactor(reg, names))
