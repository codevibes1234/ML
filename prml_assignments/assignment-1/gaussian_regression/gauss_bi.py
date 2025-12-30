import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, tol=1e-4):
    indices = np.array([i for i in range(k)])
    centroids = X[indices]
    
    while True:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids
    
    return centroids

path = '../data/dataset-2/train100.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)

N = data.shape[0]
K = N//10

inp = np.array([[row[0],row[1]] for row in data])
t = [row[2] for row in data]

centroids = kmeans(np.array(inp),K)

sigma = 8

design_matrix = []

for i in range(N):
    row = []
    for j in range(K):
        row.append(np.exp(-(np.linalg.norm(inp[i]-centroids[j]))**2 / (2*sigma**2)))
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

#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111, projection='3d')
#ax.set_title('3D Gaussian Plane Fitting with K = N/20')
#ax.scatter(inp[:,0], inp[:,1], t, color='blue', s=20, label='True outputs')
#ax.plot_trisurf(inp[:,0], inp[:,1], predicted_output, color='red', alpha=0.3)  
#ax.set_xlabel('x1 Data')
#ax.set_ylabel('x2 Data')
#ax.set_zlabel('y')

#from matplotlib.patches import Patch

#plane_patch = Patch(facecolor='red', edgecolor='r', alpha=0.3, label='Predicted plane')
#ax.legend(handles=[ax.collections[0], plane_patch])
#plt.savefig("d2_kn20_test_gauss.png")

plt.scatter(t,predicted_output,color="blue",marker="o",s=20,label="Points")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
x_line = np.linspace(min(t), max(t), 100)  # range covering x
plt.plot(x_line, x_line, "r--", label="y = x")
plt.legend()
plt.savefig("d2b_train_gauss.png")