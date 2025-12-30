import numpy as np
import analysis as anl
from Naive_Bayes import naive_bayes as nb
from Naive_Bayes import parameters as pms
from GMM import gmm
import plot_level as pl
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.lines import Line2D
import plot

path_train = "data/dataset-3/train.csv"
path_test = "data/dataset-3/test.csv"
path_val = "data/dataset-3/val.csv"

# label_path_train = "data/dataset-3/train_label.csv"
# label_path_test = "data/dataset-3/test_label.csv"
# label_path_val = "data/dataset-3/val_label.csv" 

NUM_CLASSES = 5
NUM_FEATURES = 36
Q_LIST = [2,3,4,5]
COV_TYPES = ['full','diag']

def print_data(true_labels_train:np.ndarray,true_labels_test:np.ndarray,true_labels_val:np.ndarray,output_labels_train:np.ndarray,output_labels_test:np.ndarray,output_labels_val:np.ndarray,cov_type,q):
    with open("results.txt","a") as f:
        print(f"{cov_type} {q}",file = f)
        output_labels_train = output_labels_train.astype(int)
        output_labels_test = output_labels_test.astype(int)
        output_labels_val = output_labels_val.astype(int)

        confusion_matrix_train = anl.make_confusion_matrix(true_labels_train,output_labels_train,NUM_CLASSES)
        confusion_matrix_test = anl.make_confusion_matrix(true_labels_test,output_labels_test,NUM_CLASSES)
        confusion_matrix_val = anl.make_confusion_matrix(true_labels_val,output_labels_val,NUM_CLASSES)
        print("confusion matrices:",file=f)
        print("train:",file=f)
        print(confusion_matrix_train,file=f)
        print("test:",file=f)
        print(confusion_matrix_test,file=f)
        print("val:",file = f)
        print(confusion_matrix_val,file=f)

        train_accuracy = anl.find_accuracy(confusion_matrix_train)
        test_accuracy = anl.find_accuracy(confusion_matrix_test)
        val_accuracy = anl.find_accuracy(confusion_matrix_val)
        print("Accuracy:",file=f)
        print("Train:",file=f)
        print(train_accuracy,file=f)
        print("Test:",file=f)
        print(test_accuracy,file=f)
        print("val:",file=f)
        print(val_accuracy,file=f)

        train_precision = anl.find_precision_vector(confusion_matrix_train)
        test_precision = anl.find_precision_vector(confusion_matrix_test)
        val_precision = anl.find_precision_vector(confusion_matrix_val)
        print("Precision:",file=f)
        print("Train:",file=f)
        print(train_precision,file=f)
        print("Test:",file=f)
        print(test_precision,file=f)
        print("val:",file=f)
        print(val_precision,file=f)

        train_recall = anl.find_recall_vector(confusion_matrix_train)
        test_recall = anl.find_recall_vector(confusion_matrix_test)
        val_recall = anl.find_recall_vector(confusion_matrix_val)
        print("Recall:",file=f)
        print("Train:",file=f)
        print(train_recall,file=f)
        print("Test:",file=f)
        print(test_recall,file=f)
        print("val:",file=f)
        print(val_recall,file=f)

        train_F1 = anl.find_F1_vector(train_precision,train_recall)
        test_F1 = anl.find_F1_vector(test_precision,test_recall)
        val_F1 = anl.find_F1_vector(val_precision,val_recall)
        print("F1:",file=f)
        print("Train:",file=f)
        print(train_F1,file=f)
        print("Test:",file=f)
        print(test_F1,file=f)
        print("val:",file=f)
        print(val_F1,file=f)

        avg_train_precision = np.average(train_precision)
        avg_test_precision = np.average(test_precision)
        avg_val_precision = np.average(val_precision)
        print("Avg precision:",file=f)
        print("Train:",file=f)
        print(avg_train_precision,file=f)
        print("Test:",file=f)
        print(avg_test_precision,file=f)
        print("val:",file=f)
        print(avg_val_precision,file=f)

        avg_train_recall = np.average(train_recall)
        avg_test_recall = np.average(test_recall)
        avg_val_recall = np.average(val_recall)
        print("Avg recall:",file=f)
        print("Train:",file=f)
        print(avg_train_recall,file=f)
        print("Test:",file=f)
        print(avg_test_recall,file=f)
        print("val:",file=f)
        print(avg_val_recall,file=f)

        avg_train_F1 = np.average(train_F1)
        avg_test_F1 = np.average(test_F1)
        avg_val_F1 = np.average(val_F1)
        print("Avg F1:",file=f)
        print("Train:",file=f)
        print(avg_train_F1,file=f)
        print("Test:",file=f)
        print(avg_test_F1,file=f)
        print("val:",file=f)
        print(avg_val_F1,file=f)

data_train = np.genfromtxt(path_train,delimiter=',',skip_header=1)
true_labels_train = data_train[:,NUM_FEATURES]
# true_labels_train = np.genfromtxt(label_path_train,delimiter=',',skip_header=1)
data_train_list = []
n = true_labels_train.shape[0]
for i in range(NUM_CLASSES):
    lst = []
    for j in range(n):
        if true_labels_train[j] == i:
            lst.append(data_train[j])
    data_train_list.append(lst)
for i in range(NUM_CLASSES):
    data_train_list[i] = np.array(data_train_list[i])
for i in range(NUM_CLASSES):
    data_train_list[i] = data_train_list[i][:,0:NUM_FEATURES]
data_train = data_train[:,0:NUM_FEATURES]
data_test = np.genfromtxt(path_test,delimiter=',',skip_header=1)
true_labels_test = data_test[:,NUM_FEATURES]
# true_labels_test = np.genfromtxt(label_path_test,delimiter=',',skip_header=1)
data_test = data_test[:,0:NUM_FEATURES]
data_val = np.genfromtxt(path_val,delimiter=',',skip_header=1)
true_labels_val = data_val[:,NUM_FEATURES]
# true_labels_val = np.genfromtxt(label_path_val,delimiter=',',skip_header=1)
data_val = data_val[:,0:NUM_FEATURES]

true_labels_train = true_labels_train.reshape(1,-1)
true_labels_test = true_labels_test.reshape(1,-1)
true_labels_val = true_labels_val.reshape(1,-1)
true_labels_train = true_labels_train.astype(int)
true_labels_test = true_labels_test.astype(int)
true_labels_val = true_labels_val.astype(int)

# mean_vector_list = pms.find_mean_vector_list(data_train,true_labels_train,NUM_CLASSES,NUM_FEATURES)
# cov_matrix_list = pms.find_cov_matrix_list(data_train,true_labels_train,mean_vector_list,NUM_CLASSES,NUM_FEATURES)
prior_probabs = pms.calculate_prior_probab(true_labels_train,NUM_CLASSES)

# output_labels_train = nb.naive_bayes(data_train,cov_matrix_list,mean_vector_list,prior_probabs,NUM_CLASSES,NUM_FEATURES)
# output_labels_test = nb.naive_bayes(data_test,cov_matrix_list,mean_vector_list,prior_probabs,NUM_CLASSES,NUM_FEATURES)
# output_labels_val = nb.naive_bayes(data_val,cov_matrix_list,mean_vector_list,prior_probabs,NUM_CLASSES,NUM_FEATURES)
 
# print_data(true_labels_train,true_labels_test,true_labels_val,output_labels_train,output_labels_test,output_labels_val)

for cov_type in COV_TYPES:
    for q in Q_LIST:
        estimators = gmm.get_estimators(data_train_list,NUM_CLASSES,cov_type,q)
        output_labels_train = gmm.find_labels(data_train,estimators,prior_probabs,NUM_CLASSES)
        output_labels_test = gmm.find_labels(data_test,estimators,prior_probabs,NUM_CLASSES)
        output_labels_val = gmm.find_labels(data_val,estimators,prior_probabs,NUM_CLASSES)
        print_data(true_labels_train,true_labels_test,true_labels_val,output_labels_train,output_labels_test,output_labels_val,cov_type,q)

        # x = np.linspace(-5, 5, 200)
        # y = np.linspace(-5, 5, 200)
        # X, Y = np.meshgrid(x, y)
        # fig,ax = plt.subplots(figsize=(7,6))
        # colors = ['blue','red']
        # XY = np.column_stack([X.ravel(), Y.ravel()])
        # # levels = chi2.ppf([0.25,0.5,0.75],df=2)
        # for i in range(NUM_CLASSES):
        #     # mean_vector = mean_vector_list[i]
        #     # cov_matrix = cov_matrix_list[i]
        #     # cov_matrix_inverse = np.linalg.inv(cov_matrix)
        #     # def mahalanobis_dist(data):
        #     #     diff_vector = data - mean_vector
        #     #     diff_vector_transpose = np.transpose(diff_vector)
        #     #     return (diff_vector@cov_matrix_inverse)@diff_vector_transpose
        #     # pl.plot_level(ax,mahalanobis_dist,X,Y,levels,colors[i])
        #     log_probs = estimators[i].score_samples(XY)  
        #     probs = np.exp(log_probs)
        #     Z = probs.reshape(X.shape) 
        #     ax.contour(X,Y,Z,levels=3,colors=colors[i])
        # for i in range(NUM_CLASSES):
        #     data_train_list[i] = np.array(data_train_list[i])
        #     points_x = data_train_list[i][:,0]
        #     points_y = data_train_list[i][:,1]
        #     ax.scatter(points_x,points_y,color=colors[i],marker='o',s=5)
        # legend_elements = [
        #     Line2D([0], [0], color='blue', lw=2, label='Class 0'),
        #     Line2D([0], [0], color='red', lw=2, label='Class 1')
        # ]
        # ax.legend(handles=legend_elements)
        # ax.set_xlabel('Feature 1')
        # ax.set_ylabel('Feature 2')
        # ax.set_title('Level curves with training data points')
        # plt.savefig(f'{cov_type}_{q}.png')
        # def find_labels(data:np.ndarray):
        #     data = np.array(data)
        #     posterior_probabs = np.empty(2,dtype=object)
        #     for i in range(2):
        #         posterior_probabs[i] = estimators[i].score_samples(data.reshape(1,-1))
        #     max_label = 0
        #     max_log_probab = posterior_probabs[0] + np.log(prior_probabs[0])
        #     for i in range(1,2):
        #         log_probab = posterior_probabs[i] + np.log(prior_probabs[i])
        #         if log_probab > max_log_probab:
        #             max_label = i
        #             max_log_probab = log_probab
        #     return max_label
# def naive_bayes(data):  
#     data = np.array(data)
#     max_label = 0
#     max_probab = nb.calculate_class_cond_probab(data,cov_matrix_list[0],mean_vector_list[0],2) * prior_probabs[0]
#     for j in range(1,2):
#         probab = nb.calculate_class_cond_probab(data,cov_matrix_list[j],mean_vector_list[j],2) * prior_probabs[j]
#         if probab > max_probab:
#             max_probab = probab
#             max_label = j
#     return max_label
# plot.show(data_train,naive_bayes)