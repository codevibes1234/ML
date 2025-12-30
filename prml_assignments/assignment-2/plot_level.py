import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2

def plot_level(data_train, funcs, color = ['red','blue','green']):
    _L = int(np.max(data_train[:, -1])) + 1
    _d = 2

    true_labels_train = data_train[:,_d]
    data_train_list = []
    n = true_labels_train.shape[0]
    for i in range(_L):
        lst = []
        for j in range(n):
            if true_labels_train[j] == i:
                lst.append(data_train[j])
        data_train_list.append(lst)
    for i in range(_L):
        data_train_list[i] = np.array(data_train_list[i])
    for i in range(_L):
        data_train_list[i] = data_train_list[i][:,0:2]
    for i in range(_L):
        data_train_list[i] = np.array(data_train_list[i])
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    fig,ax = plt.subplots(figsize=(7,6))
    Z = np.empty_like(X)
    levels = chi2.ppf([0.25,0.5,0.75],df=2)
    for i in range(_L):
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                data_point = np.array([X[row,col],Y[row,col]])
                Z[row,col] = funcs(data_point, i)
        ax.contour(X,Y,Z,levels=levels,colors=color[i])
    for i in range(_L):
        points_x = data_train_list[i][:,0]
        points_y = data_train_list[i][:,1]
        ax.scatter(points_x,points_y,color=color[i],marker='o',s=5)
    #legend_elements = [
    #    Line2D([0], [0], color=color[0], lw=2, label='Class 0'),
    #    Line2D([0], [0], color=color[1], lw=2, label='Class 1')
    #]
    legend_elements = [Line2D([0], [0], color=color[i], lw=2, label="Class %d" % (i + 1)) for i in range(_L)]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Level curves with training data points')
    plt.savefig('fig.png')
