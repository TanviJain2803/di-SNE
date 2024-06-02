#!/usr/bin/env python

# functions used in diSNE's main function

import warnings
warnings.filterwarnings('ignore') # suppress warning messages while running
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import hdf5plugin

# perform grid search to obtain values of sigma based on perplexity (used to get similarities)
def search_sigma(distances, curr, perplexity):
    """
    Obtain σ's based on user's perplexity value.

    Parameters:
        distances (np.ndarray): Array containing the pairwise distances between data points.
        curr (int): Index of current data point.
        perplexity (int): User-specified perplexity value.

    Returns:
        sig (float): The value of σ that satisfies the perplexity condition.
    """
#     print("perplexity:", perplexity)
    
    result = np.inf  # Set first result to be infinity

    norm = np.linalg.norm(distances, axis=1)
    std_norm = np.std(norm)  # Use standard deviation of norms to define search space

    for sig_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
        # Equation 1 Numerator
        p = np.exp(-(norm**2) / (2 * sig_search**2))

        # Set p = 0 when i = j
        p[curr] = 0

        # Equation 1 (ε -> 0)
        ε = np.nextafter(0, 1)
        p_new = np.maximum(p / np.sum(p), ε)

        # Handle potential NaNs
        if np.any(np.isnan(p_new)):
           # print(f"Skipping sig_search {sig_search}: p_new contains NaNs")
            continue

        # Shannon Entropy
        p_new = p_new[p_new > 0]  # Avoid log2(0) by filtering out non-positive values
        if len(p_new) == 0:  # Check if p_new is empty after filtering
           # print(f"Skipping sig_search {sig_search}: p_new is empty after filtering")
            continue

        H = -np.sum(p_new * np.log2(p_new))

        # Handle potential NaN in H
        if np.isnan(H):
          #  print(f"Skipping sig_search {sig_search}: H is NaN, p_new: {p_new}")
            continue

        # Get log(perplexity equation) as close to equality
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sig = sig_search

    return sig
    
# compute affinities matrix for X in the original (high-dimensional) space
def get_highdim_affinities(X, perplexity):
    """
    Function to obtain similarities matrix in original high-dimensional space.

    Parameters:
    X (np.ndarray): Input dataset
    perplexity (int): Perplexity of the joint probability distribution

    Returns:
    P (np.ndarray of shape (number of samples * (num samples - 1) / 2)): Joint probabilities matrix.
    """ 
#     n = len(X)
    n = X.shape[0]
    P = np.zeros((n, n))
    
    for i in range(0, n):
        # equation 1 numerator
#         print("X[" + str(i) + "]")
#         print(X[i])
        difference = X[i] - X
        sig_i = search_sigma(difference, i, perplexity) # call search function to get sigma
        norm = np.linalg.norm(difference, axis=1)
        P[i, :] = np.exp(-(norm**2) / (2 * sig_i**2))

        # Set p = 0 when j = i
        np.fill_diagonal(P, 0)

        # compute equation 1
        P[i, :] = P[i, :] / np.sum(P[i, :])

    # Set 0 values to minimum numpy value (ε approx. = 0)
    eps = np.nextafter(0, 1)
    P = np.maximum(P, eps)

    print("Completed Pairwise Affinities Matrix. \n")

    return P

# convert original affinities matrix into joint probabilities (symmetric) affinities matrix
def convert_to_jointprob(P):
    """
    Obtain symmetric affinities matrix from original affinities matrix to be utilized in t-SNE.

    Parameters:
    P (np.ndarray): Input (original) affinity matrix.

    Returns:
    P_symmetric (np.ndarray): Symmetric affinities matrix.

    """
#     n = len(P)
    n = P.shape[0]
    P_symmetric = np.zeros(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            P_symmetric[i, j] = (P[i, j] + P[j, i]) / (2 * n)

    # Set 0 values to minimum numpy value (ε approx. = 0)
    eps = np.nextafter(0, 1)
    P_symmetric = np.maximum(P_symmetric, eps)

    return P_symmetric

# sample initial solution in lower-dimensional space
def initialize(X, n_dim: int = 2, initialization: str = "random"):
    """
    Create initial solution for t-SNE either randomly or using PCA.

    Parameters:
        X (np.ndarray): The input data array.
        n_dimensions (int): The number of dimensions for the output solution. Default is 2.
        initialization (str): The initialization method. Can be 'random' or 'PCA'. Default is 'random'.

    Returns:
        soln (np.ndarray): The initial solution for t-SNE.

    Raises:
        ValueError: If the initialization method is neither 'random' nor 'PCA'.
    """

    # sample initial solution 
    if initialization == "random" or initialization != "PCA":
#         soln = np.random.normal(loc=0, scale=1e-4, size=(len(X), n_dim))
        soln = np.random.normal(loc=0, scale=1e-4, size=(X.shape[0], n_dim))
    elif initialization == "PCA":
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered)
        soln = X_centered @ Vt.T[:, :n_dimensions]
    else:
        raise ValueError("Initialization must be 'random' or 'PCA'")

    return soln

# compute affinity matrix in lower-dimensional space
def get_lowdim_affinities(Y):
    """
    Obtain low-dimensional affinity matrix, uses a Student t-distribution with 1 degree of freedom.

    Parameters:
    Y (np.ndarray): Low-dimensional representation of the data points.

    Returns:
    Q (np.ndarray): The low-dimensional affinities matrix.
    """

#     n = len(Y)
    n = Y.shape[0]
    Q = np.zeros(shape=(n, n))

    for i in range(0, n):
        # equation 4 numerator
        difference = Y[i] - Y
        norm = np.linalg.norm(difference, axis=1)
        Q[i, :] = (1 + norm**2) ** (-1)

    # Set p = 0 when j = i
    np.fill_diagonal(Q, 0)

    # equation 4
    Q = Q / Q.sum()

    # Set 0 values to minimum numpy value (ε approx. = 0)
    eps = np.nextafter(0, 1)
    Q = np.maximum(Q, eps)

    return Q

# compute gradient of the cost function 
def compute_gradient(P, Q, Y):
    """
    Obtain gradient of cost function at current point Y.
    The cost function is the Kullback-Leibler divergence between the joint probability distributions in high dimensional 
    vs low dimensional space

    Parameters:
    P (np.ndarray): The higher-dimension joint probability distribution matrix.
    Q (np.ndarray): The lower-dimenion Student t-distribution matrix.
    Y (np.ndarray): The current point in the low-dimensional space.

    Returns:
    gradient (np.ndarray): The gradient of the cost function at the current point Y.
    """

#     n = len(P)
    n = P.shape[0]

    # Compute gradient
    gradient = np.zeros(shape=(n, Y.shape[1]))
    for i in range(0, n):
        difference = Y[i] - Y
        A = np.array([(P[i, :] - Q[i, :])])
        B = np.array([(1 + np.linalg.norm(difference, axis=1)) ** (-1)])
        C = difference
        gradient[i] = 4 * np.sum((A * B).T * C, axis=0)

    return gradient

# save t-SNE results to file "diSNE-results" in current working directory
def save_data_to_h5ad(adata, filename = "diSNE-results"):
    """
    Save data to an H5AD file format with specified compression settings.

    Parameters:
        adata (AnnData): The AnnData object containing the data.
        filename (str), OPTIONAL: The name of the file to which the data will be saved, default is diSNE-results
    """
    compression = hdf5plugin.FILTERS["zstd"]
    compression_opts = hdf5plugin.Zstd(clevel=5).filter_options

    adata.write_h5ad(filename, compression=compression, compression_opts=compression_opts)
    print("Updated AnnData object saved as file", filename)
    
# generate plot of diSNE results and save it to current directory as the file name specified
def plot_results(adata, filename='diSNE_plot.png', title='diSNE results', figsize=(10, 8)):
    tsne_out = adata.obsm['X_tsne']
    labels = adata.obs['leiden'].astype(int).values
    plt.figure(figsize=figsize)
    scatter = plt.scatter(tsne_out[:, 0], tsne_out[:, 1], c=labels, cmap='viridis', marker='o')
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
#     plt.colorbar()
    plt.show()
    plt.savefig(filename) 
    print("Plot saved as", filename)
#     return plt
      
# save the plot results to the given directory
# def save_tsne_plot(adata, feature='leiden', filename = './tsne_plot.png', title='diSNE results', figsize=(10, 8)):
# def save_tsne_plot(adata, filename='./tsne_plot.png', title='diSNE results', figsize=(10, 8)):
#     plot = plot_results(adata, 'leiden', title, figsize)
#     plot.savefig(filename)  # Save the plot to a file
#     print("Plot saved as", filename)