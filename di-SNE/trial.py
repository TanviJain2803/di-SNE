
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Step 1: Compute pairwise distances
def pairwise_distances(X):
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
    Q = inv_distances / np.sum(inv_distances)
    Q = np.maximum(Q, 1e-12)
    return Q

# Step 5: Compute gradients and update embedding
def compute_gradients(P, Q, Y):
    n, dim = Y.shape
    dY = np.zeros((n, dim))
    PQ_diff = P - Q
    for i in range(n):
        dY[i, :] = 4 * np.sum((PQ_diff[:, i] * (Y[i, :] - Y)), axis=0)
    return dY

def update_embedding(Y, dY, learning_rate=200.0):
    return Y - learning_rate * dY

# Step 6: The main t-SNE function
def tsne(X, perplexity=30.0, n_iter=1000, learning_rate=200.0):
    distances = pairwise_distances(X)
    P = compute_joint_probabilities(distances, perplexity)
    Y = initialize_embedding(X.shape[0])
    
    for iter in range(n_iter):
        Q = compute_low_dim_affinities(Y)
        dY = compute_gradients(P, Q, Y)
        Y = update_embedding(Y, dY, learning_rate)
        
        if iter % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter}: cost = {cost}")
    
    return Y

# Example usage
# Assuming 'data' is your preprocessed single-cell RNA-seq data
# data = pd.read_csv('your_single_cell_data.csv').values
# result = tsne(data, perplexity=30.0, n_iter=1000)

# Assuming you have a 'cluster' column in your original data
# Add t-SNE results and cluster labels to a new DataFrame for plotting
tsne_df['Cluster'] = data['cluster']

import matplotlib.pyplot as plt

def plot_tsne_results(Y, labels=None, title='t-SNE Visualization'):
    """
    Plot the t-SNE results.

    Parameters:
    - Y: np.ndarray, the t-SNE embedding, shape (n_samples, 2)
    - labels: np.ndarray or list, cluster labels for the data points, shape (n_samples,)
    - title: str, title of the plot
    """
    plt.figure(figsize=(10, 8))
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            plt.scatter(
                Y[labels == label, 0], Y[labels == label, 1],
                label=f'Cluster {label}', alpha=0.6
            )
        plt.legend()
    else:
        plt.scatter(Y[:, 0], Y[:, 1], alpha=0.6)
    
    plt.title(title)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.show()

# Example usage
# Assuming 'result' is the output of the tsne function and 'clusters' are the cluster labels
# clusters = data['cluster'].values  # or any array of cluster labels
# plot_tsne_results(result, labels=clusters)
