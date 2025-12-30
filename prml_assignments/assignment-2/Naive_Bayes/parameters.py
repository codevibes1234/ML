import numpy as np

def find_mean_vector_list(data_train,true_labels,num_classes,num_features):
    data_train = np.array(data_train)
    n = data_train.shape[0]
    mean_vector_list = np.zeros((num_classes,num_features))
    num_points = np.zeros(num_classes)
    for i in range(n):
        for j in range(num_features):
            mean_vector_list[true_labels[0][i]][j] += data_train[i][j]; 
        num_points[true_labels[0][i]] += 1
    for i in range(num_classes):
        for j in range(num_features):
            mean_vector_list[i][j] /= num_points[i]
    return mean_vector_list

def find_cov_matrix(data_train,true_labels,label,mean_vector,num_features):
    data_train = np.array(data_train)
    n = data_train.shape[0]
    cov_matrix = np.zeros((num_features,num_features))
    num_points = 0
    for i in range(n):
        if true_labels[0][i] == label:
            num_points += 1
            for j in range(num_features):
                cov_matrix[j][j] += (data_train[i][j]-mean_vector[j])**2
    for i in range(num_features):
        cov_matrix[i][i] /= num_points
    return cov_matrix

def find_cov_matrix_list(data_train,true_labels,mean_vector_list,num_classes,num_features):
    cov_matrix_list = np.zeros((num_classes,num_features,num_features))
    for i in range(num_classes):
        cov_matrix_list[i] = find_cov_matrix(data_train,true_labels,i,mean_vector_list[i],num_features)
    return cov_matrix_list

def calculate_prior_probab(true_labels,num_class):
    true_labels = np.array(true_labels)
    n = true_labels.shape[1]
    probab = np.zeros(num_class)
    for i in range(n):
        probab[true_labels[0][i]] += 1
    for i in range(num_class):
        probab[i] /= n
    return probab