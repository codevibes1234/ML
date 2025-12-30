import sys
sys.path.append("../")

import numpy as np
import math
import performance  # custom module for printing the performance 
import plot

dataset = 1
toPlot = 1  # >0 for plotting 
'''
K = []
if dataset != 3:
    K = [1, 5, 9]
else:
    K = [1, 9, 15]
'''
K = 1

arr = np.genfromtxt("../data/dataset-%d/train.csv" % (dataset), delimiter=',', skip_header=1) # the last column contains the class labels
d = np.size(arr, axis=1) - 1            # dimension of the feature vector
L = int(np.max(arr[:, -1])) + 1         # no.of classes

def dist(arr1, arr2): # since the last element is a class label, it is excluded while calculating the distance
    x = 0
    for i in range(0, d):
        x += (arr1[i] - arr2[i]) ** 2
    x = math.sqrt(x)
    return x

def KNN_helper(X):
    l = [] # stores the distances and class labels
    for n in range(0, np.size(arr, axis=0)):
        temp = []
        temp.append(dist(X, arr[n]))
        temp.append(arr[n][d])
        l.append(temp)
    l = np.array(l)
    l = l[np.argsort(l[:, 0])]

    k = [0 for _col in range(0, L)]  # count
    z = [0 for _col in range(0, L)]  # total dist
    for i in range(0, K):
        class_label = np.int64(l[i][1]) # used as index to access the elements
        k[class_label] += 1
        z[class_label] += l[i][0]
    for i in range(0, L):
        if (k[i] != 0):
            z[i] /= -k[i]        # stores the negative of the distance
    indices = np.lexsort((z, k)) # sort by k first, and then by z and return the indices that would sort the data
    return indices[-1]           # so, indices[-1] would be the predicted label

def KNN(data_arr):
    confusion_mat = [[0 for _col in range(0, L)] for _row in range(0, L)]
    for n in range(0, np.size(data_arr, axis=0)):
        true_label = int(data_arr[n][d])
        predicted_label = KNN_helper(data_arr[n])
        confusion_mat[true_label][predicted_label] += 1

    performance.eval(confusion_mat)

train_data = np.genfromtxt("../data/dataset-%d/train.csv" % (dataset), delimiter=',', skip_header=1)
val_data = np.genfromtxt("../data/dataset-%d/val.csv" % (dataset), delimiter=',', skip_header=1)
test_data = np.genfromtxt("../data/dataset-%d/test.csv" % (dataset), delimiter=',', skip_header=1)

print("For K = %d:" % (K))
print("Train data:")
KNN(train_data)

print("\nValidation data:")
KNN(val_data)

print("\nTest data:")
KNN(test_data)
print("---\n")

if dataset != 3 and toPlot > 0:
    plot.show(arr, KNN_helper)
