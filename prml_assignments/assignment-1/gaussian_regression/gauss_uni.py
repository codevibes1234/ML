import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, tol=1e-4):
    indices = np.array([i for i in range(k)])
    centroids = X[indices]
    
    while True:
        distances = np.abs(X[:, np.newaxis] - centroids[np.newaxis, :])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean() for i in range(k)])
        if np.all(np.linalg.norm(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    
    return centroids

path = '../data/dataset-1/test.txt'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_sorted = data[np.argsort(data[:, 0])]

N = data.shape[0]
K = N//10

first_col = [row[0] for row in data_sorted]
t = [row[1] for row in data_sorted]

centroids = kmeans(np.array(first_col),K)

sigma = 8

design_matrix = []

for i in range(N):
    row = []
    for j in range(K):
        row.append(np.exp(-(first_col[i]-centroids[j])**2 / (2*sigma**2)))
    design_matrix.append(row)

design_matrix_transpose = np.transpose(design_matrix)
pseudo_inv = design_matrix_transpose @ design_matrix
pseudo_inv = np.linalg.inv(pseudo_inv) @ design_matrix_transpose
t_transpose = np.transpose(t)
arg_mat = pseudo_inv @ t_transpose

predicted_output = design_matrix @ arg_mat

erms = np.linalg.norm(predicted_output-t)
import math
erms = erms / (math.sqrt(N))

print(erms)

#from scipy.interpolate import make_interp_spline

#smooth_pred = make_interp_spline(first_col, predicted_output)(first_col)
#plt.figure(figsize=(8,5))
#plt.title('Gaussian Curve fitting with K = 4')
#plt.xlabel('Input')
#plt.ylabel('Output')
#plt.scatter(first_col,t,color='blue', label='True outputs')
#plt.plot(first_col, smooth_pred, color='red', label='Predicted outputs (curve)')
#plt.legend()
#plt.savefig("d1_k4_t10_gauss.png")

plt.scatter(t,predicted_output,color="blue",marker="o",s=20,label="Points")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
x_line = np.linspace(min(t), max(t), 100)  # range covering x
plt.plot(x_line, x_line, "r--", label="y = x")
plt.legend()
plt.savefig("d1_test_gauss.png")