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

path = '../data/dataset-3/dataset3_val.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)

N = data.shape[0]
K = N//20

inp = np.array([[row[0],row[1],row[2]] for row in data])
t1,t2,t3 = [row[3] for row in data], [row[4] for row in data], [row[5] for row in data]

centroids = kmeans(np.array(inp),K)

sigma = 0.75

design_matrix = []

for i in range(N):
    row = []
    for j in range(K):
        row.append(np.exp(-(np.linalg.norm(inp[i]-centroids[j]))**2 / (2*sigma**2)))
    design_matrix.append(row)

design_matrix_transpose = np.transpose(design_matrix)
pseudo_inv = design_matrix_transpose @ design_matrix
pseudo_inv = np.linalg.inv(pseudo_inv) @ design_matrix_transpose
t1_transpose,t2_transpose,t3_transpose = np.transpose(t1), np.transpose(t2), np.transpose(t3)
arg_mat1,arg_mat2,arg_mat3 = pseudo_inv @ t1_transpose, pseudo_inv @ t2_transpose, pseudo_inv @ t3_transpose

predicted_output1,predicted_output2,predicted_output3 = design_matrix @ arg_mat1,design_matrix @ arg_mat2,design_matrix @ arg_mat3

erms1,erms2,erms3 = np.linalg.norm(predicted_output1-t1),np.linalg.norm(predicted_output2-t2),np.linalg.norm(predicted_output3-t3)
import math
erms1,erms2,erms3 = erms1/(math.sqrt(N)),erms2/(math.sqrt(N)),erms3/(math.sqrt(N))

print(erms1,erms2,erms3)

#fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#axs[0,0].scatter(t1,predicted_output1,color="blue",marker="o",s=20,label="Points")
#axs[0,0].set_xlabel("Actual Values")
#axs[0,0].set_ylabel("Predicted Values")
#axs[0,0].set_title("Output1")
#x_line = np.linspace(min(t1), max(t1), 100)  # range covering x
#axs[0,0].plot(x_line, x_line, "r--", label="y = x")
#axs[0,0].legend()
#axs[0,1].scatter(t2,predicted_output2,color="blue",marker="o",s=20,label="Points")
#axs[0,1].set_xlabel("Actual Values")
#axs[0,1].set_ylabel("Predicted Values")
#axs[0,1].set_title("Output2")
#x_line = np.linspace(min(t2), max(t2), 100)  # range covering x
#axs[0,1].plot(x_line, x_line, "r--", label="y = x")
#axs[0,1].legend()
#axs[1,0].scatter(t3,predicted_output3,color="blue",marker="o",s=20,label="Points")
#axs[1,0].set_xlabel("Actual Values")
#axs[1,0].set_ylabel("Predicted Values")
#axs[1,0].set_title("Output3")
#x_line = np.linspace(min(t3), max(t3), 100)  # range covering x
#axs[1,0].plot(x_line, x_line, "r--", label="y = x")
#axs[1,0].legend()
#axs[1, 1].axis('off')
#plt.suptitle("Actual vs Predicted")
#plt.tight_layout()
#plt.savefig("d3_test.png")