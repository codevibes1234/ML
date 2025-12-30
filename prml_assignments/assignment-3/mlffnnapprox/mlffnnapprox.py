from sklearn.neural_network import MLPRegressor
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

hidden_layer_sizes = [(8,),(15,10)]
num_features = [2,3]

dataset = int(input("Enter dataset number: "))
plot_num = int(input("Enter plot number: "))

path_train = f"../data/dataset-{dataset}/train.csv" 
path_val = f"../data/dataset-{dataset}/val.csv"
path_test = f"../data/dataset-{dataset}/test.csv" 

dataset -= 1

data_train = np.genfromtxt(path_train,delimiter=',',skip_header=1)
data_val = np.genfromtxt(path_val,delimiter=',',skip_header=1)
data_test = np.genfromtxt(path_test,delimiter=',',skip_header=1)

inp_train = data_train[:,0:num_features[dataset]]
true_train = data_train[:,num_features[dataset]:]
inp_val = data_val[:,0:num_features[dataset]]
true_val = data_val[:,num_features[dataset]:]
inp_test = data_test[:,0:num_features[dataset]]
true_test = data_test[:,num_features[dataset]:]

def scatter_plot(true,predicted,name):
    num_outputs = 1
    if predicted.ndim > 1:
        num_outputs = predicted.shape[1]
    for i in range(num_outputs):
        plt.figure()
        if num_outputs != 1:
            plt.scatter(true[:,i],predicted[:,i],color="blue",marker="o",s=10,label="Points")
        else:
            plt.scatter(true,predicted,color="blue",marker="o",s=10,label="Points")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        if num_outputs != 1:
            x_line = np.linspace(min(true[:,i]), max(true[:,i]), 100) 
        else:
            x_line = np.linspace(min(true), max(true), 100)
        plt.plot(x_line, x_line, "r--", label="y = x")
        plt.legend()
        plt.savefig(f"2_{dataset+1}_{i}_{name}.png")
        plt.close()

def plot(xh,yh,output,name):
    xh_lin = np.linspace(xh.min(), xh.max(), 50)
    yh_lin = np.linspace(yh.min(), yh.max(), 50)
    X_grid, Y_grid = np.meshgrid(xh_lin, yh_lin)
    zh = output.reshape(xh.shape)
    Z_grid = griddata((xh, yh), zh, (X_grid, Y_grid), method='cubic')
    Z_grid = Z_grid.reshape(X_grid.shape)
    print(X_grid.shape,Y_grid.shape,Z_grid.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Output')
    if num_epochs != 10000:
        ax.set_title(f'Surface at Epoch {num_epochs} for {name}')
        name = name.replace(" ","_")
        plt.savefig(f'3_1_{num_epochs}_{name}.png')
    else:
        ax.set_title(f'Surface at convergence for {name}')
        name = name.replace(" ","_")
        plt.savefig(f'3_1_conv_{name}.png')
    plt.close()

if plot_num == 1:
    approximator = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes[dataset],activation='tanh',solver='sgd',learning_rate_init=0.01,tol=1e-3,max_iter=10000,random_state=42)
    approximator.fit(inp_train,true_train)
    print(approximator.loss_)
    plt.figure()
    plt.plot(approximator.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.title("Training Error vs Epoch Plot")
    plt.savefig(f'1_{dataset+1}')
    plt.close()

elif plot_num == 2:
    approximator = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes[dataset],activation='tanh',solver='sgd',learning_rate_init=0.01,tol=0.001,max_iter=10000,random_state=42)
    approximator.fit(inp_train,true_train)
    predicted_train = approximator.predict(inp_train)
    predicted_test = approximator.predict(inp_test)
    scatter_plot(true_train,predicted_train,"train")
    scatter_plot(true_test,predicted_test,"test")

else:
    num_epochs = int(input("Enter the number of epochs: "))
    approximator = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes[dataset],activation='tanh',solver='sgd',learning_rate_init=0.01,tol=0.001,max_iter=num_epochs,random_state=42)
    approximator.fit(inp_train,true_train)
    w1,b1 = approximator.coefs_[0], approximator.intercepts_[0]
    w2,b2 = approximator.coefs_[1], approximator.intercepts_[1]
    hidden_output1 = np.tanh(inp_train @ w1[:,0]+b1[0])
    hidden_output2 = np.tanh(inp_train @ w1[:,1]+b1[1])
    # print(hidden_output1,hidden_output2)
    output = (inp_train @ w1 + b1) @ w2 + b2
    xh = inp_train[:,0]
    yh = inp_train[:,1]
    plot(xh,yh,hidden_output1,"Hidden node 1")
    plot(xh,yh,hidden_output2,"Hidden node 2")
    plot(xh,yh,output,"Output Node")  