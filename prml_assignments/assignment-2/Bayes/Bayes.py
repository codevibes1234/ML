import sys
sys.path.append("../")

import numpy as np
import math
import performance
import plot
import plot_level

dataset = 1  # to pick the dataset
part = 1     # 0 for part-a and 1 for part-b
toPlot = 1   # >0 for plotting

arr = np.genfromtxt("../data/dataset-%d/train.csv" % (dataset), delimiter=',', skip_header=1)
N = np.size(arr, axis=0)         # total no.of examples
d = np.size(arr, axis=1) - 1     # dimension of the feature vector
L = int(np.max(arr[:, -1])) + 1  # total no.of classes

count = np.array([0 for l in range(0, L)]) # stores the no.of examples in each class
mean = np.array([[0. for i in range(0, d)] for l in range(0, L)])  # stores the mean vector for each class
for n in range(0, N):
    l = int(arr[n][d])
    count[l] += 1
    mean[l] += arr[n][:d] 
for l in range(0, L):
    if count[l] != 0:
        mean[l] /= count[l]

cov = np.array([[[0. for j in range(0, d)] for i in range(0, d)] for l in range(0, L)]) # stores the covariance matrix for each class
for n in range(0, N):
    l = int(arr[n][d])
    X = arr[n][:d] - mean[l]
    X = np.reshape(X, shape=(d,1))
    cov[l] += X @ (X.T)
for l in range(0, L):
    if count[l] != 0:
        cov[l] /= count[l]

if part == 0:
    avg_cov = np.array([[0. for j in range(0, d)] for i in range(0, d)])
    for l in range(0, L):
        avg_cov += cov[l]
    avg_cov /= L
    for l in range(0, L):
        cov[l] = avg_cov

det = [0. for l in range(0, L)] # stores the discriminant value for the covariance matrix for each class
for l in range(0, L):
    det[l] = np.linalg.det(cov[l])

cov_inv = [] # stores the inverse of the covariance matrix for each class
for l in range(0, L):
    cov_inv.append(np.linalg.inv(cov[l]))
cov_inv = np.array(cov_inv)

def squared_mahalanobis_dist(x, l):
    X = x - mean[l]
    X = np.reshape(X, shape=(d,1))
    return 0.5 * (X.T @ cov_inv[l] @ X)

def Bayes_helper(x): # returns the predicted class label for a single example
    posterior_probab = [] # unnormalized values
    for l in range(0, L):
        X = x - mean[l]
        X = np.reshape(X, shape=(d,1))
        value = (-0.5 * math.log(det[l])) + (-1.0 * squared_mahalanobis_dist(x, l)) + math.log(count[l])
        posterior_probab.append(value)
    posterior_probab = np.array(posterior_probab)
    return np.argmax(posterior_probab)

def Bayes(data_arr):
    confusion_mat = [[0 for j in range(0, L)] for i in range(0, L)]
    for n in range(0, np.size(data_arr, axis=0)):
        true_label = int(data_arr[n][d])
        predicted_label = Bayes_helper(data_arr[n][:d])
        confusion_mat[true_label][predicted_label] += 1

    performance.eval(confusion_mat)

train_data = np.genfromtxt("../data/dataset-%d/train.csv" % (dataset), delimiter=',', skip_header=1)
val_data = np.genfromtxt("../data/dataset-%d/val.csv" % (dataset), delimiter=',', skip_header=1)
test_data = np.genfromtxt("../data/dataset-%d/test.csv" % (dataset), delimiter=',', skip_header=1)

print("Train data:")
Bayes(train_data)

print("\nValidation data:")
Bayes(val_data)

print("\nTest data:")
Bayes(test_data)

if dataset != 3 and toPlot > 0:
    plot.show(arr, Bayes_helper)
    plot_level.plot_level(arr, squared_mahalanobis_dist)
