import numpy as np
import math
import matplotlib.pyplot as plt

def create_design_matrix(data_array, M):
    N = len(data_array) # number of examples
    design_matrix = []

    for n in range(0, N):
        x = data_array[n][0]
        basis_vector = []
        basis_fun = 1
        for i in range(0, M + 1):
            basis_vector.append(basis_fun)
            basis_fun *= x
        design_matrix.append(basis_vector)
    
    return np.array(design_matrix)

def get_Erms(y, t):
    N = len(y)
    error = 0
    for i in range(0, N):
        error += (y[i] - t[i]) ** 2
    error = math.sqrt(error / N)
    return error

# load training data
#train_data_file_path = "../data/dataset-1/train10.txt"
train_data_file_path = "../data/dataset-1/train50.txt"
train_data_array = np.genfromtxt(train_data_file_path, delimiter=',', skip_header=1)

# load validation data
val_data_file_path = "../data/dataset-1/val.txt"
val_data_array = np.genfromtxt(val_data_file_path, delimiter=',', skip_header=1)
val_target_matrix = val_data_array[:, 1]

# load test data
test_data_file_path = "../data/dataset-1/test.txt"
test_data_array = np.genfromtxt(test_data_file_path, delimiter=',', skip_header=1)
test_target_matrix = test_data_array[:, 1]

for M in [3, 5, 7, 9]:
    Erms_data = [] # stores Erms of train, validation, and test data for different values of hyperparameters
    lambda_values = [0.001, 0.1, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, reg_param in enumerate(lambda_values):
        # training phase
        train_design_matrix = create_design_matrix(train_data_array, M)

        mat_trans = np.transpose(train_design_matrix) # transpose of train design matrix
        XTX = mat_trans @ train_design_matrix
        XTX_reg = XTX + reg_param * np.eye(XTX.shape[0])

        target_matrix = train_data_array[:, 1]
        parameter_vector_reg = np.linalg.solve(XTX_reg, mat_trans @ target_matrix)
        parameter_vector_noreg = np.linalg.solve(XTX, mat_trans @ target_matrix)

        model_output_reg = train_design_matrix @ parameter_vector_reg
        model_output_noreg = train_design_matrix @ parameter_vector_noreg
       
        # plot for training phase
        axes[i].scatter(train_data_array[:, 0], target_matrix, color="red", label="Train Data")
        axes[i].plot(train_data_array[:, 0], model_output_reg, color="green", label="With Regularization")
        axes[i].plot(train_data_array[:, 0], model_output_noreg, color="blue", label="No Regularization")
        axes[i].set_xlabel("X Data")
        axes[i].set_ylabel("Y Output")
        axes[i].legend()
        axes[i].set_title("Degree=%d, Lambda=%f" % (M, reg_param))

        # validation phase
        val_design_matrix = create_design_matrix(val_data_array, M)
        val_model_output = val_design_matrix @ parameter_vector_reg

        # test phase
        test_design_matrix = create_design_matrix(test_data_array, M)
        test_model_output = test_design_matrix @ parameter_vector_reg

        # evaluating Erms values
        Erms_train = get_Erms(model_output_reg, target_matrix)
        Erms_val = get_Erms(val_model_output, val_target_matrix)
        Erms_test = get_Erms(test_model_output, test_target_matrix)
        Erms = [Erms_train, Erms_val, Erms_test]
        Erms_data.append(Erms)

    Erms_data = np.array(Erms_data)
    for row in Erms_data:
        print(row)
    axes[-1].axis("off") # hides the last (empty) subplot
    plt.tight_layout()
    plt.show()
