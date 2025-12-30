import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_design_matrix(data_array, M):
    N = len(data_array) # number of examples
    design_matrix = []

    for n in range(0, N):
        x1 = data_array[n][0]
        x2 = data_array[n][1]
        x1_basis = []
        x2_basis = []
        x1_curr = 1
        x2_curr = 1
        for i in range(0, M + 1):
            x1_basis.append(x1_curr)
            x2_basis.append(x2_curr)
            x1_curr *= x1
            x2_curr *= x2
        basis_vector = []
        for i in range(0, M + 1):
            for j in range(0, M - i + 1):
                basis_vector.append(x1_basis[i] * x2_basis[j])
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
#train_data_file_path = "../data/dataset-2/train25.csv"
train_data_file_path = "../data/dataset-2/train100.csv"
train_data_array = np.genfromtxt(train_data_file_path, delimiter=',', skip_header=1)

# load validation data
val_data_file_path = "../data/dataset-2/val.csv"
val_data_array = np.genfromtxt(val_data_file_path, delimiter=',', skip_header=1)
val_target_matrix = val_data_array[:, 2]

# load test data
test_data_file_path = "../data/dataset-2/test.csv"
test_data_array = np.genfromtxt(test_data_file_path, delimiter=',', skip_header=1)
test_target_matrix = test_data_array[:, 2]

Erms_data = [] # stores Erms of train, validation, and test data for different values of hyperparameters
M_values = [2, 4, 6, 8]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection':'3d'})
axes = axes.ravel()

for i, M in enumerate(M_values):
    # training phase
    train_design_matrix = create_design_matrix(train_data_array, M)

    mat_trans = np.transpose(train_design_matrix) # transpose of train design matrix
    XTX = mat_trans @ train_design_matrix

    target_matrix = train_data_array[:, 2]
    parameter_vector = np.linalg.solve(XTX, mat_trans @ target_matrix)

    model_output = train_design_matrix @ parameter_vector
   
    # plot for training phase
    x1_grid = np.linspace(train_data_array[:, 0].min(), train_data_array[:, 0].max(), 50)
    x2_grid = np.linspace(train_data_array[:, 1].min(), train_data_array[:, 1].max(), 50)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)

    mesh_points = np.column_stack((X1_mesh.ravel(), X2_mesh.ravel()))
    mesh_design_matrix = create_design_matrix(mesh_points, M)
    mesh_predictions = mesh_design_matrix @ parameter_vector
    Z_mesh = mesh_predictions.reshape(X1_mesh.shape)
    
    axes[i].scatter(train_data_array[:, 0], train_data_array[:, 1], target_matrix, color="red", s=10, marker='o', label="Train Data")
    axes[i].plot_surface(X1_mesh, X2_mesh, Z_mesh, color="blue", alpha=0.7)
    axes[i].set_xlabel("X1 Data")
    axes[i].set_ylabel("X2 Data")
    axes[i].set_zlabel("Y Output")
    axes[i].legend()
    axes[i].set_title("3D Polynomial Curve Fitting Degree=%d" % M)

    # validation phase
    val_design_matrix = create_design_matrix(val_data_array, M)
    val_model_output = val_design_matrix @ parameter_vector

    # test phase
    test_design_matrix = create_design_matrix(test_data_array, M)
    test_model_output = test_design_matrix @ parameter_vector

    # evaluating Erms values
    Erms_train = get_Erms(model_output, target_matrix)
    Erms_val = get_Erms(val_model_output, val_target_matrix)
    Erms_test = get_Erms(test_model_output, test_target_matrix)
    Erms = [Erms_train, Erms_val, Erms_test]
    Erms_data.append(Erms)

Erms_data = np.array(Erms_data)
for row in Erms_data:
    print(row)
plt.tight_layout()
plt.show()
