# 2. Use a KMeans regression model with the Iris data set. Graph the fit when using differing numbers of
# clusters. Graph the result and either corroborate or refute the assumption that the data set represents
# 3 different varieties of iris.

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def createIrisPred(X, y, n_clusters):
    return KMeans(n_clusters).fit_predict(X, y)


def createPlots(X, y):
    subplot = 221
    for i in range(2, 6):
        y_pred = createIrisPred(X, y, i)
        plt.subplot(subplot)
        subplot += 1
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title(f"Iris Prediction with {i} Clusters")
    plt.show()


if __name__ == "__main__":
    names = load_iris().feature_names
    X, y = load_iris(return_X_y=True)
    createPlots(X, y)
