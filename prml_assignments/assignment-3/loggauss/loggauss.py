from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/venkatks/Code/git_repositories/prml-assignments/assignment-3")
import latex as ltx

num_features = 36
k = int(input("Enter K: "))
sigma = float(input("Enter Sigma: "))
plot = int(input("Plot?: "))

path_train = "../data/dataset-4/train.csv" 
path_val = "../data/dataset-4/val.csv"
path_test = "../data/dataset-4/test.csv" 

data_train = np.genfromtxt(path_train,delimiter=',',skip_header=1)
data_val = np.genfromtxt(path_val,delimiter=',',skip_header=1)
data_test = np.genfromtxt(path_test,delimiter=',',skip_header=1)

inp_train = data_train[:,0:num_features]
true_train = data_train[:,num_features:]
inp_val = data_val[:,0:num_features]
true_val = data_val[:,num_features:]
inp_test = data_test[:,0:num_features]
true_test = data_test[:,num_features:]

def kmeans(X,tol=1e-4):
    indices = np.array([i for i in range(k)])
    centroids = X[indices]    
    while True:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids    
    return centroids

def create_design_matrix(inp):
    centroids = kmeans(inp)
    design_matrix = []
    for i in range(inp.shape[0]):
        row = []
        for j in range(k):
            row.append(np.exp(-(np.linalg.norm(inp[i]-centroids[j]))**2 / sigma))
        design_matrix.append(row)
    return design_matrix

design_matrix_train = np.array(create_design_matrix(inp_train))
design_matrix_val = np.array(create_design_matrix(inp_val))
design_matrix_test = np.array(create_design_matrix(inp_test))

classifier = MLPClassifier(hidden_layer_sizes=(),solver='sgd',learning_rate_init=0.01,tol=1e-3,max_iter=10000,random_state=100,alpha=0)
classifier.fit(design_matrix_train,true_train)

if plot:
    plt.figure()
    plt.plot(classifier.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.title("Training Error vs Epoch Plot")
    plt.savefig(f"loss_{k}_{sigma}.png")
    plt.close()

pred_train = classifier.predict(design_matrix_train)
pred_test = classifier.predict(design_matrix_test)

train_accuracy = accuracy_score(true_train,pred_train)
test_accuracy = accuracy_score(true_test,pred_test)

confusion_matrix_train = confusion_matrix(true_train,pred_train)
confusion_matrix_test = confusion_matrix(true_test,pred_test)

with open(f"latex_{k}_{sigma}_accuracy.txt","w") as f:
    print(ltx.print_accuracies(train_accuracy,test_accuracy),file=f)

with open(f"latex_{k}_{sigma}_confusion_matrix.txt","w") as f:
    print(ltx.print_confusion_matrix(confusion_matrix_train,confusion_matrix_test),file=f)