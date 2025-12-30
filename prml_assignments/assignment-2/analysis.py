import numpy as np

def make_confusion_matrix(true_labels,predicted_labels,num_class):
    true_labels = np.array(true_labels)
    confusion_matrix = np.zeros((num_class,num_class))
    n = true_labels.shape[1]
    for i in range(n):
        confusion_matrix[true_labels[0][i]][predicted_labels[i]] += 1
    return confusion_matrix

def find_accuracy(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    num_class = confusion_matrix.shape[0]
    num_points = 0
    correctly_predicted_points = 0
    for i in range(num_class):
        for j in range(num_class):
            num_points += confusion_matrix[i][j]
            if i == j:
                correctly_predicted_points += confusion_matrix[i][j]
    return correctly_predicted_points/num_points

def find_precision_vector(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    num_class = confusion_matrix.shape[0]
    precision_vector = np.zeros(num_class)
    for i in range(num_class):
        correctly_predicted = confusion_matrix[i][i]
        tot_predicted = np.sum(confusion_matrix[:,i])
        precision_vector[i] = correctly_predicted/tot_predicted
    return precision_vector

def find_recall_vector(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    num_class = confusion_matrix.shape[0]
    recall_vector = np.zeros(num_class)
    for i in range(num_class):
        correctly_predicted = confusion_matrix[i][i]
        tot_predicted = np.sum(confusion_matrix[i])
        recall_vector[i] = correctly_predicted/tot_predicted
    return recall_vector

def find_F1_vector(precision_vector,recall_vector):
    precision_vector = np.array(precision_vector)
    recall_vector = np.array(recall_vector)
    num_class = recall_vector.shape[0]
    F1_vector = np.zeros(num_class)
    for i in range(num_class):
        F1_vector[i] = (2*precision_vector[i]*recall_vector[i])/(precision_vector[i]+recall_vector[i])
    return F1_vector