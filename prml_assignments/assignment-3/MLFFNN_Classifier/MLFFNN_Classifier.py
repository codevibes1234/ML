import sys
sys.path.append("../")

from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import numpy as np
import plot_decision_regions
import plot_surface
import plot_error_curve
import latex

task = 3  # for this assignment use only 3 or 4
plot = 1  # >0 will plot 

data = np.genfromtxt("../data/dataset-%d/train.csv" % (task), delimiter=',', skip_header=1)
samples = np.delete(data, -1, axis=1) # deletes the last column in data array
labels = data[:, -1]

hidden_layers = ()
if task == 3:
    hidden_layers = (12, 8)
elif task == 4:
    hidden_layers = (25, 15)

clf, max_epochs = None, []
if task == 3:
    max_epochs = [1, 10, 50, 1000]
else:
    max_epochs = [1000]

# configuring the model
for epochs in max_epochs:
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='tanh', solver='sgd', alpha=.0, learning_rate_init=0.7, max_iter=epochs, random_state=75, tol=0.001, momentum=0.9)
    clf.fit(samples, labels)

    if task == 3 and plot:
        plot_surface.show(data, clf)

train_data = np.genfromtxt("../data/dataset-%d/train.csv" % (task), delimiter=',', skip_header=1)
val_data = np.genfromtxt("../data/dataset-%d/val.csv" % (task), delimiter=',', skip_header=1)
test_data = np.genfromtxt("../data/dataset-%d/test.csv" % (task), delimiter=',', skip_header=1)

input_data = [train_data, val_data, test_data]

confusion_matrices = []
accuracies = []
for data_arr in input_data:
    input_samples = np.delete(data_arr, -1, axis=1)
    input_labels = np.astype(data_arr[:, -1], int)

    predicted_labels = clf.predict(input_samples)
    
    confusion_matrices.append(metrics.confusion_matrix(input_labels, predicted_labels))
    accuracies.append(clf.score(input_samples, input_labels))

print(latex.print_accuracies(accuracies[0], accuracies[2]))
print(latex.print_confusion_matrix(confusion_matrices[0], confusion_matrices[1]))

if task == 3 and plot:
    plot_decision_regions.show(data, clf)
if plot:
    plot_error_curve.show(clf)
