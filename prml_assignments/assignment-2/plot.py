import matplotlib.pyplot as plt
import numpy as np

def show(arr, classifier):
    _d = 2
    _N = np.size(arr, axis=0)
    _L = int(np.max(arr[:, -1])) + 1

    x = [[] for l in range(0, _L)]
    y = [[] for l in range(0, _L)]
    for n in range(0, _N):
        l = int(arr[n][_d])
        x[l].append(arr[n][0])
        y[l].append(arr[n][1])

    for l in range(0, _L):
        plt.scatter(x[l], y[l], s=10, marker='D')

    xmin = np.min(arr[:, 0]) - 1
    xmax = np.max(arr[:, 0]) + 1
    ymin = np.min(arr[:, 1]) - 1
    ymax = np.max(arr[:, 1]) + 1

    samples = 200
    xx = np.linspace(xmin, xmax, samples)
    yy = np.linspace(ymin, ymax, samples)
    X, Y = np.meshgrid(xx, yy)
    Z = []
    for i in range(0, samples):
        _temp = []
        for j in range(0, samples):
            _temp.append(classifier(np.array([X[i][j], Y[i][j]])))
        Z.append(_temp)
    Z = np.array(Z)
    plt.scatter(X, Y, s=2, c=Z, alpha=0.1)
    plt.show()
    plt.close()
