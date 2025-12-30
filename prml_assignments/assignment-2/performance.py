import numpy as np

def eval(confusion_mat):
    confusion_mat = np.array(confusion_mat)
    L = np.size(confusion_mat, axis=0)
    N = np.sum(confusion_mat)
    
    accuracy = 0
    for i in range(0, L):
        accuracy += confusion_mat[i][i]
    accuracy /= N
    print("accuracy:", end=" ")
    print(accuracy)
    print("confusion matrix:")
    print(confusion_mat)

    precision = []
    recall = []
    F1 = []
    for i in range(0, L):
        p = confusion_mat[i][i]
        col = np.sum(confusion_mat[:, i])
        p /= col

        r = confusion_mat[i][i]
        row = np.sum(confusion_mat[i])
        r /= row

        f1 = (2 * p * r) / (p + r)

        precision.append(p)
        recall.append(r)
        F1.append(f1)

    print("precision:", end=" ")
    print(*precision)
    print("   recall:", end=" ")
    print(*recall)
    print("       F1:", end=" ")
    print(*F1)

    print("average precision:", end=" ")
    print(sum(precision) / L)
    print("   average recall:", end=" ")
    print(sum(recall) / L)
    print("       average F1:", end=" ")
    print(sum(F1) / L)
