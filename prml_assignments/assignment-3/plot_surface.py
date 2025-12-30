import matplotlib.pyplot as plt
import numpy as np

def soft_max(arr):
    return arr / np.sum(np.exp(arr))

def show(data, clf):
    x = data[:, 0]
    y = data[:, 1]

    x_min = np.min(x) - 1
    x_max = np.max(x) + 1
    y_min = np.min(y) - 1
    y_max = np.max(y) + 1
    
    no_of_samples = 300

    xx = np.linspace(x_min, x_max, num=no_of_samples)
    yy = np.linspace(y_min, y_max, num=no_of_samples)
    X, Y = np.meshgrid(xx, yy)

    first_layer_output = []   # stores the output of the first and second nodes in the first hidden layer of each example
    second_layer_output = []  # stores the output of the first and second nodes in the second hidden layer of each example
    final_layer_output = []   # stores the output of the first and second nodes in the output layer of each example
    for i in range(no_of_samples):
        for j in range(no_of_samples):
            _output1 = np.tanh(clf.coefs_[0].T @ np.array([[X[i][j]], [Y[i][j]]]) + np.reshape(clf.intercepts_[0], shape=(12,1)))
            _output2 = np.tanh(clf.coefs_[1].T @ _output1 + np.reshape(clf.intercepts_[1], shape=(8,1)))
            _output = soft_max(clf.coefs_[2].T @ _output2 + np.reshape(clf.intercepts_[2], shape=(1, 1)))

            first_layer_output.append([_output1[0][0], _output1[1][0]])
            second_layer_output.append([_output2[0][0], _output2[1][0]])
            final_layer_output.append([_output[0][0]])

    layer_output = [first_layer_output, second_layer_output, final_layer_output]
    for i, output in enumerate(layer_output):
        output = np.array(output)
        if i != 2:
            fig, ax = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
            ax[0].plot_surface(X, Y, np.reshape(output[:, 0], shape=(no_of_samples,no_of_samples)), cmap='jet')
            ax[1].plot_surface(X, Y, np.reshape(output[:, 1], shape=(no_of_samples,no_of_samples)), cmap='jet')
            ax[0].set_title("Output at hidden layer %d: node 1" % (i + 1), fontsize=8.5)
            ax[1].set_title("Output at hidden layer %d: node 2" % (i + 1), fontsize=8.5)
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
            ax.plot_surface(X, Y, np.reshape(output[:, 0], shape=(no_of_samples,no_of_samples)), cmap='jet')
            ax.set_title("Output at output layer: node 1", fontsize=10)
        plt.show()
        plt.close()
