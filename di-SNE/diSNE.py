import argparse
# packages to cluster data
import sys
import os
sys.path.append(os.environ["HOME"]+"/.local/lib/python3.9/site-packages")
import scanpy as sc, anndata as ad
import numpy as np
import harmonypy #?
import leidenalg
import pandas as pd
from sklearn.datasets import make_blobs #?
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Step 1: Compute pairwise distances
def pairwise_distances(X):
    if issparse(X):
        X = X.toarray() 
    return squareform(pdist(X, 'euclidean'))

# Step 2: Compute joint probabilities P in high-dimensional space
def compute_joint_probabilities(distances, perplexity=30.0):
    (n, _) = distances.shape
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        H, thisP = Hbeta(Di, beta[i])
        
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > 1e-5 and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf:
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == -np.inf:
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)
    return P

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

# Step 3: Initialize embedding Y
def initialize_embedding(n, dim=2):
    return np.random.randn(n, dim)

# Step 4: Compute low-dimensional affinities Q
def compute_low_dim_affinities(Y):
    distances = pairwise_distances(Y)
    inv_distances = 1 / (1 + distances)
    np.fill_diagonal(inv_distances, 0)
    inv_distances_sum = np.sum(inv_distances)
    Q = inv_distances / (inv_distances_sum + 1e-12)  # Add epsilon to avoid division by zero
    Q = np.maximum(Q, 1e-12)
    return Q

# Step 5: Compute gradients and update embedding
def compute_gradients(P, Q, Y):
    n, dim = Y.shape
    dY = np.zeros((n, dim))
    PQ_diff = P - Q
    for i in range(n):
        dY[i, :] = 4 * np.sum((PQ_diff[:, i].reshape(-1, 1) * (Y[i, :] - Y)), axis=0)
    return dY

def update_embedding(Y, dY, learning_rate=200.0):
    # Normalize the gradients
    grad_norm = np.linalg.norm(dY, axis=1, keepdims=True)
    dY = dY / (grad_norm + 1e-12)
    
    # Clip the gradients to prevent excessively large updates
    max_step_size = 1.0
    dY = np.clip(dY, -max_step_size, max_step_size)
    
    return Y - learning_rate * dY

# Step 6: The main t-SNE function modified to store results in adata.obsm['X_tsne']
def tsne(adata, perplexity=30.0, n_iter=1000, learning_rate=10.0):
    X = adata.X
    distances = pairwise_distances(X)
    P = compute_joint_probabilities(distances, perplexity)
    Y = initialize_embedding(X.shape[0])
    
    for iter in range(n_iter):
        Q = compute_low_dim_affinities(Y)
        dY = compute_gradients(P, Q, Y)
        Y = update_embedding(Y, dY, learning_rate)
        
        if iter % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            # Uncomment the next line if you want to see the progress
#             print(f"Iteration {iter}: cost = {cost}")
            pass
    
    # Store the resulting t-SNE embedding in adata.obsm['X_tsne']
    adata.obsm['X_tsne'] = Y
    return

def plot_tsne_results(adata, feature, title='t-SNE results', figsize=(10, 8)):
    tsne_out = adata.obsm['X_tsne']
    labels = adata.obs[feature].astype(int).values
    plt.figure(figsize=figsize)
    scatter = plt.scatter(tsne_out[:, 0], tsne_out[:, 1], c=labels, cmap='viridis', marker='o')
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        prog="di-SNE",
        description="Command-line tool to perform t-SNE"
    )
    
    # input 
    parser.add_argument("data", help="Annotated data matrix (?)", type=str)
    
    # output 
    
    # other options
    # perplexity
    parser.add_argument("-p", "--perplexity", 
                        help="Perplexity value used in tSNE. It is recommended to use a larger perplexity for larger datasets. Default=30, recommended range: 5-50", 
                        type=float) #int?
    # learning rate
    parser.add_argument("-r", "--learning-rate", 
                        help="Learning rate used in tSNE, default=200. Recommend range: 100-1000",
                        type=float) #int?
    
    # parse args
    args = parser.parse_args()

#dataset = args.data
#perplexity = args.perplexity
#learning_rate = args.learning_rate

    # run tsne with user inputs
    tsne(dataset) #Running tsne
    
    # plot the t-SNE results
#     plot_tsne_results(combined_var, feature='leiden')
    
   if __name__ == "__main__":
       main()
    
