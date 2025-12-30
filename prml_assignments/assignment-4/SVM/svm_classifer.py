import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
import sys
sys.path.append("/home/venkatks/Code/git_repositories/prml-assignments")
import latex as ltx
import matplotlib.pyplot as plt

dataset = int(input("Enter the dataset number: "))
plot = int(input("Should I plot? "))
NUM_FEATURES = [2,2,36]
C = [1,10,100]
degrees = [2,4,7,9]
widths = [0.5,0.7,1,10]

path_train = f"../data/dataset-{dataset}/train.csv"
path_val = f"../data/dataset-{dataset}/val.csv"
path_test = f"../data/dataset-{dataset}/test.csv"

data_train = np.genfromtxt(path_train,delimiter=',',skip_header=1)
data_val = np.genfromtxt(path_val,delimiter=',',skip_header=1)
data_test = np.genfromtxt(path_test,delimiter=',',skip_header=1)

num_features = NUM_FEATURES[dataset-1]

inp_train,true_train = data_train[:,0:num_features],data_train[:,num_features:]
inp_val,true_val = data_val[:,0:num_features],data_val[:,num_features:]
inp_test,true_test = data_test[:,0:num_features],data_test[:,num_features:]

def print_accuracies_and_confusion_matrices(clf,f):
    pred_train = clf.predict(inp_train)
    pred_val = clf.predict(inp_val)
    pred_test = clf.predict(inp_test)
    train_accuracy = accuracy_score(true_train,pred_train)
    val_accuracy = accuracy_score(true_val,pred_val)
    test_accuracy = accuracy_score(true_test,pred_test)
    confusion_matrix_train = confusion_matrix(true_train,pred_train)
    confusion_matrix_test = confusion_matrix(true_test,pred_test)
    accuracies = ltx.print_accuracies(train_accuracy,val_accuracy,test_accuracy)
    confusion_matrices = ltx.print_confusion_matrix(confusion_matrix_train,confusion_matrix_test)
    print(accuracies,file=f)
    print(confusion_matrices,file=f)

def plot_decision_region(clf):
    x_min, x_max = inp_train[:, 0].min() - 1, inp_train[:, 0].max() + 1
    y_min, y_max = inp_train[:, 1].min() - 1, inp_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(inp_train[:, 0], inp_train[:, 1], c=true_train, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=120, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Regions with Support Vectors")
    plt.legend()
    plt.savefig("1.png")
    plt.close()

def plot_decision_regions_with_bounded_and_unbounded(clf,part):
    c = C[dataset-1]
    x_min, x_max = inp_train[:, 0].min() - 1, inp_train[:, 0].max() + 1
    y_min, y_max = inp_train[:, 1].min() - 1, inp_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    alphas = np.abs(clf.dual_coef_).flatten()
    eps = 1e-6
    bounded_mask = np.isclose(alphas, c, atol=eps)
    unbounded_mask = (alphas > eps) & (alphas < c - eps)
    bounded_sv = clf.support_vectors_[bounded_mask]
    unbounded_sv = clf.support_vectors_[unbounded_mask]
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(inp_train[:, 0], inp_train[:, 1], c=true_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.scatter(bounded_sv[:, 0], bounded_sv[:, 1],
                s=40, facecolors='none', edgecolors='red', linewidths=2,
                label='Bounded SVs')
    plt.scatter(unbounded_sv[:, 0], unbounded_sv[:, 1],
                s=40, facecolors='none', edgecolors='green', linewidths=2,
                label='Unbounded SVs')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Regions with Bounded/Unbounded Support Vectors")
    plt.legend()
    plt.savefig(f"2{part}.png")
    plt.close()

if dataset == 1:
    with open("1.txt","w") as f:
        clf = SVC(kernel='linear')
        clf.fit(inp_train,true_train)        
        print_accuracies_and_confusion_matrices(clf,f)
    if plot == 1:
        plot_decision_region(clf)
elif dataset == 2:        
    for deg in degrees:
        for c in C:
            with open(f"2_{deg}_{c}.txt","w") as f:
                clf = SVC(kernel='poly',degree=deg,C=c)
                clf.fit(inp_train,true_train)
                print_accuracies_and_confusion_matrices(clf,f)
    for width in widths:
        for c in C:
            with open(f"2_{width}_{c}.txt","w") as f:
                clf = SVC(kernel='rbf',gamma=width,C=c)
                clf.fit(inp_train,true_train)
                print_accuracies_and_confusion_matrices(clf,f)
    if plot == 1:
        clf1 = SVC(kernel='poly',degree=7,C=10)
        clf1.fit(inp_train,true_train)
        clf2 = SVC(kernel='rbf',gamma=1,C=1)
        clf2.fit(inp_train,true_train)
        plot_decision_regions_with_bounded_and_unbounded(clf1,"a")
        plot_decision_regions_with_bounded_and_unbounded(clf2,"b")
else:
    for deg in degrees:
        for c in C:
            with open(f"3_{deg}_{c}.txt","w") as f:
                clf = SVC(kernel='poly',degree=deg,C=c)
                clf.fit(inp_train,true_train)
                print_accuracies_and_confusion_matrices(clf,f)
    for width in widths:
        for c in C:
            with open(f"3_{width}_{c}.txt","w") as f:
                clf = SVC(kernel='rbf',gamma=width,C=c)
                clf.fit(inp_train,true_train)
                print_accuracies_and_confusion_matrices(clf,f)

# def print_bounded_and_unbounded(clf,part):
#     alphas = abs(clf.dual_coef_).flatten()
#     eps = 1e-6
#     bounded = np.isclose(alphas,clf.C,atol=eps)
#     unbounded = (alphas > eps) & (alphas < clf.C - eps)
#     n_bounded = bounded.sum()
#     n_unbounded = unbounded.sum()
#     n_total = len(alphas)
#     pct_bounded = 100 * n_bounded / n_total
#     pct_unbounded = 100 * n_unbounded / n_total
#     prnt = ltx.bounded_and_unbounded(pct_bounded,pct_unbounded)
#     with open(f"bounded_and_unbounded{part}.txt","w") as f:
#         print(prnt,file=f)

# clf1 = SVC(kernel='poly',degree=2,C=1)
# clf2 = SVC(kernel='rbf',gamma=10,C=1)
# clf1.fit(inp_train,true_train)
# clf2.fit(inp_train,true_train)
# print_bounded_and_unbounded(clf1,"a")
# print_bounded_and_unbounded(clf2,"b")