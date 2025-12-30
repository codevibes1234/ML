import sys
sys.path.append("../")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as metrics
import numpy as np
import plot_decision_regions
import plot_error_curve
import latex

plot = 1  # >0 will plot

train_data = np.genfromtxt("../data/dataset-3/train.csv", delimiter=',', skip_header=1)
val_data = np.genfromtxt("../data/dataset-3/val.csv", delimiter=',', skip_header=1)
test_data = np.genfromtxt("../data/dataset-3/test.csv", delimiter=',', skip_header=1)

degrees = [5, 7, 9]
for M in degrees:
    poly = PolynomialFeatures(M)
    samples = poly.fit_transform(np.delete(train_data, -1, axis=1))
    labels = train_data[:, -1]

    clf = MLPClassifier(hidden_layer_sizes=(), solver='sgd', alpha=0., learning_rate_init=0.7, max_iter=1000, random_state=42, tol=0.001, momentum=0.9)
    clf.fit(samples, labels)

    input_data = [train_data, val_data, test_data]
    confusion_matrices = []
    accuracies = []
    for data_arr in input_data:
        input_samples = np.delete(data_arr, -1, axis=1)
        input_labels = data_arr[:, -1]

        poly = PolynomialFeatures(M)
        input_samples = poly.fit_transform(input_samples)

        predicted_labels = clf.predict(input_samples)

        confusion_matrices.append(metrics.confusion_matrix(input_labels, predicted_labels))
        accuracies.append(clf.score(input_samples, input_labels))

    print(latex.print_accuracies(accuracies[0], accuracies[2]))
    print(latex.print_confusion_matrix(confusion_matrices[0], confusion_matrices[2]))

    if plot:
        plot_decision_regions.show(train_data, clf, 1, M)
        plot_error_curve.show(clf)
