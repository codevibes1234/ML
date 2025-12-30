import numpy as np
import math

def calculate_class_cond_probab(data,cov_matrix,mean_vector,num_features):
    diff_vector = data - mean_vector
    diff_vector_transpose = np.transpose(diff_vector)
    cov_matrix_inverse = np.linalg.inv(cov_matrix)
    probab = np.exp(-((diff_vector@cov_matrix_inverse)@diff_vector_transpose)/2)
    det_cov_matrix = np.linalg.det(cov_matrix)
    probab = probab/(math.sqrt(det_cov_matrix))
    probab = probab/(math.pow(2*math.pi,num_features/2))
    return probab

def naive_bayes(data,cov_matrix_list,mean_vector_list,prior_probabs,num_classes,num_features):  
    data = np.array(data)
    n = data.shape[0]
    output_labels = np.zeros(n)
    for i in range(n):
        max_label = 0
        max_probab = calculate_class_cond_probab(data[i],cov_matrix_list[0],mean_vector_list[0],num_features) * prior_probabs[0]
        for j in range(1,num_classes):
            probab = calculate_class_cond_probab(data[i],cov_matrix_list[j],mean_vector_list[j],num_features) * prior_probabs[j]
            if probab > max_probab:
                max_probab = probab
                max_label = j
        output_labels[i] = max_label
    return output_labels