import numpy as np
import math
import matplotlib.pyplot as plt

def create_design_matrix(data_array, M):
    N = len(data_array) # number of examples
    design_matrix = []

    for n in range(0, N):
        x1 = data_array[n][0]
        x2 = data_array[n][1]
        x3 = data_array[n][2]

        x1_basis = []
        x2_basis = []
        x3_basis = []

        x1_curr = 1
        x2_curr = 1
        x3_curr = 1
        for i in range(0, M + 1):
            x1_basis.append(x1_curr)
            x2_basis.append(x2_curr)
            x3_basis.append(x3_curr)

            x1_curr *= x1
            x2_curr *= x2
            x3_curr *= x3
        
        basis_vector = []
        for i in range(0, M + 1):
            for j in range(0, M - i + 1):
                for k in range(0, M - i - j + 1):
                    basis_vector.append(x1_basis[i] * x2_basis[j] * x3_basis[k])
        design_matrix.append(basis_vector)
    
    return np.array(design_matrix)

def get_Erms(y, t):
    N = len(y)
    error = 0
    for i in range(0, N):
        __error = 0
        for j in range(0, 3):
            __error += (y[i][j] - t[i][j]) ** 2
        error += __error
    error = math.sqrt(error / N)
    return error

# load training data
train_data_file_path = "../data/dataset-3/train.csv"
train_data_array = np.genfromtxt(train_data_file_path, delimiter=',', skip_header=1)
target_matrix = train_data_array[:, 3:6]

# load validation data
val_data_file_path = "../data/dataset-3/val.csv"
val_data_array = np.genfromtxt(val_data_file_path, delimiter=',', skip_header=1)
val_target_matrix = val_data_array[:, 3:6]

# load test data
test_data_file_path = "../data/dataset-3/test.csv"
test_data_array = np.genfromtxt(test_data_file_path, delimiter=',', skip_header=1)
test_target_matrix = test_data_array[:, 3:6]

for M in [2, 3, 4]:
    Erms_data = [] # stores Erms of train, validation, and test data for different values of hyperparameters
    figures = []

    for reg_param in [0.000001, 0.0001, 0.1]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        figures.append(fig)

        # training phase
        train_design_matrix = create_design_matrix(train_data_array, M)

        mat_trans = np.transpose(train_design_matrix) # transpose of train design matrix
        XTX = mat_trans @ train_design_matrix
        XTX_reg = XTX + reg_param * np.eye(XTX.shape[0])

        parameter_vector_1 = np.linalg.solve(XTX_reg, mat_trans @ target_matrix[:, 0])
        parameter_vector_2 = np.linalg.solve(XTX_reg, mat_trans @ target_matrix[:, 1])
        parameter_vector_3 = np.linalg.solve(XTX_reg, mat_trans @ target_matrix[:, 2])
        parameter_matrix = np.column_stack((parameter_vector_1, parameter_vector_2, parameter_vector_3))

        model_output = train_design_matrix @ parameter_matrix
       
        # plot for training phase
        for i in range(0, 3):
            axes[i].scatter(target_matrix[:, i], model_output[:, i], color="blue", label="Train Data")
            axes[i].axline((0, 0), slope=1, linestyle='--', color="red", label="Perfect Prediction")
            axes[i].set_xlabel("Actual Values")
            axes[i].set_ylabel("Predicted Values")
            axes[i].legend()
            axes[i].set_title("Degree=%d, Lambda=%f" % (M, reg_param))
        axes[-1].axis('off')
        fig.tight_layout()

        # validation phase
        val_design_matrix = create_design_matrix(val_data_array, M)
        val_model_output = val_design_matrix @ parameter_matrix

        # test phase
        test_design_matrix = create_design_matrix(test_data_array, M)
        test_model_output = test_design_matrix @ parameter_matrix

        # evaluating Erms values
        Erms_train = get_Erms(model_output, target_matrix)
        Erms_val = get_Erms(val_model_output, val_target_matrix)
        Erms_test = get_Erms(test_model_output, test_target_matrix)
        Erms = [Erms_train, Erms_val, Erms_test]
        Erms_data.append(Erms)

    Erms_data = np.array(Erms_data)
    for row in Erms_data:
        print(row)
    plt.show()
