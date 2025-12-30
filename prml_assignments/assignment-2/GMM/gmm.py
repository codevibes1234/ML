import numpy as np
from sklearn.mixture import GaussianMixture

def get_estimators(data_train_list,num_classes,cov_type,q):
    estimators = np.empty(num_classes,dtype=object)
    for i in range(num_classes):
        estimators[i] = GaussianMixture(n_components=q,covariance_type=cov_type)
        estimators[i].fit(data_train_list[i])
    return estimators

def find_labels(data:np.ndarray,estimators:np.ndarray,prior_probabs,num_classes):
    n = data.shape[0]
    posterior_probabs = np.empty(num_classes,dtype=object)
    for i in range(num_classes):
        posterior_probabs[i] = estimators[i].score_samples(data)
    predicted_labels = np.empty(n,dtype=int)
    for i in range(n):
        max_label = 0
        max_log_probab = posterior_probabs[0][i] + np.log(prior_probabs[0])
        for j in range(1,num_classes):
            log_probab = posterior_probabs[j][i] + np.log(prior_probabs[j])
            if log_probab > max_log_probab:
                max_label = j
                max_log_probab = log_probab
        predicted_labels[i] = max_label
    return predicted_labels