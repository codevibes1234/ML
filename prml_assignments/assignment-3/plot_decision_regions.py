from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

training_data_colors = matplotlib.colors.ListedColormap(['red', 'blue'])
region_colors = matplotlib.colors.ListedColormap(['tomato', 'deepskyblue'])

def show(data, clf, flag=0, K=0): # flag=0 : MLFFNN_Classifier, flag=1 : Logistic_Regression_Classifier and K is the degree
    x = data[:, 0]
    y = data[:, 1]

    x_min = np.min(x) - 1
    x_max = np.max(x) + 1
    y_min = np.min(y) - 1
    y_max = np.max(y) + 1

    no_of_samples = 300

    xx = np.linspace(x_min, x_max, num=no_of_samples)
    yy = np.linspace(y_min, y_max, num=no_of_samples)
    X, Y = np.meshgrid(xx, yy)

    sample_input = []
    for i in range(no_of_samples):
        for j in range(no_of_samples):
            if flag:
                poly = PolynomialFeatures(K)
                sample_input.append(np.ravel(poly.fit_transform([[X[i][j], Y[i][j]]])))
            else :
                sample_input.append([X[i][j], Y[i][j]])
    
    plt.scatter(x, y, s=10, c=data[:, -1], cmap=training_data_colors, alpha=1.0)
    plt.scatter(X, Y, s=0.3, c=clf.predict(sample_input), cmap=region_colors, alpha=0.3)
    plt.show()
    plt.close()
