import matplotlib.pyplot as plt
import numpy as np

def show(clf):
    y = clf.loss_curve_
    x = [i + 1 for i in range(len(y))]

    plt.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("Training error")
    plt.show()
    plt.close()
