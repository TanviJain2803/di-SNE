#!/usr/bin/env python

import sys
import os
import warnings
warnings.filterwarnings('ignore') # suppress warnings while running
# set up packages 
import argparse
import scanpy as sc, anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import hdf5plugin
from scipy.spatial.distance import pdist, squareform
# import functions from utils file
from . import diSNE_utils as diSNE_utils

# diSNE main algorithm
def diSNE(adata, perplexity, T, learning_rate, early_exaggeration, pca=False):
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm implementation.
    Using the results from the helper functions defined in diSNE_utils.py, this function iterates 
    and optimizes low-dimensional mapping, using a gradient descent with momentum.

    Args:
        adata (file containing AnnData object): Input file containing the AnnData object representing the user's dataset.
        perplexity (int, optional): Perplexity parameter used in algorithm. Default is 10.
        T (int, optional): Number of iterations for optimization. Default is 1000.
        learning_rate (int, optional): Learning rate for updating the low-dimensional embeddings, or controlling
        the step size at each iteration. Default is 200.
        early_exaggeration (int, optional): Factor by which the pairwise affinities are exaggerated
            during the early iterations of optimization. Default is 4.
        n_dim (int, optional): The number of dimensions of the low-dimensional embeddings. Default is 2.
        pca (bool, optional): Whether or not PCA has been run on the dataset, speeds up t-SNE

    Returns:
        dataset (AnnData object): Updated AnnData object containing diSNE results in dataset.obsm['X_tsne']

    """
    # set default values if user didn't specify
    n_dim = 2
    if perplexity is None: 
        perplexity = 10
    if T is None:
        T = 1000
    if learning_rate is None:
        learning_rate = 200
    if early_exaggeration is None:
        early_exaggeration = 4
        
    # read in user's file as an AnnData object
    dataset = ad.read_h5ad(adata)
    print("Read file", adata)
    
    if pca: # set matrix to PCA if user specified PCA option
        X = dataset.obsm['X_pca']
    else:
        X = dataset.X
    n = X.shape[0]
    
    print("Starting t-SNE algorithm")

    # Get original affinities matrix
    P = diSNE_utils.get_highdim_affinities(X, perplexity)
    P_symmetric = diSNE_utils.convert_to_jointprob(P)

    # Initialization
    Y = np.zeros(shape=(T, n, n_dim))
    Y_minus1 = np.zeros(shape=(n, n_dim))
    Y[0] = Y_minus1
    Y1 = diSNE_utils.initialize(X, n_dim)
    Y[1] = np.array(Y1)

    print("Optimizing Low Dimensional Embedding....")
    # perform optimization of the low-dimensional embedding
    for t in range(1, T - 1):
        # set momentum & early exaggeration
        if t < 250:
            momentum = 0.5
            early_exaggeration = early_exaggeration
        else:
            momentum = 0.8
            early_exaggeration = 1

        # get Low Dimensional Affinities
        Q = diSNE_utils.get_lowdim_affinities(Y[t])

        # get gradient of cost function 
        gradient = diSNE_utils.compute_gradient(early_exaggeration * P_symmetric, Q, Y[t])

        # update rule
        Y[t + 1] = Y[t] - learning_rate * gradient + momentum * (Y[t] - Y[t - 1])  # use negative gradient

        # compute current value of cost function (Kullback-Leibler divergence) and print as status update
        if t % 100 == 0:
            cost = np.sum(P_symmetric * np.log(P_symmetric / Q))
            print(f"Iteration {t} of {T}: Value of Cost Function is {cost}")

    print(
        f"Completed Low Dimensional Embedding: Final Value of Cost Function is {np.sum(P_symmetric * np.log(P_symmetric / Q))}"
    )
    soln = Y[-1]
    # update dataset's t-SNE coordinates and return dataset 
    dataset.obsm['X_tsne'] = soln
    return dataset

def main():
    # set up argument parser
    parser = argparse.ArgumentParser(
        prog="diSNE",
        description="Command-line tool to perform t-SNE (t-Distributed Stochastic Neighbor Embedding) on a pre-filtered, clustered dataset"
    )
    
    # input (REQUIRED)
    parser.add_argument("data", help="Annotated data (AnnData) matrix, with Leiden clustering already performed", type=str)
    
    # output (optional) -- output file to a specific path 
    parser.add_argument("-o", "--output", 
                        help="Path where file containing updated AnnData object with t-SNE results will be saved", type=str)
    
    # other options (OPTIONAL)
    # perplexity
    parser.add_argument("-p", "--perplexity", 
                        help="Perplexity value used in the t-SNE algorithm. It is recommended to use a larger perplexity for larger datasets. Default=10, recommended range: 5-50", 
                        type=int) 
    # learning rate
    parser.add_argument("-r", "--learning-rate", 
                        help="Learning rate used during optimization, default=200. Recommended range: 100-1000",
                        type=int)
    # number of iterations 
    parser.add_argument("-T", "--num-iterations", 
                        help="Number of iterations used for optimization, default=1000",
                        type=int) 
    # early exaggeration
    parser.add_argument("-E", "--early-exaggeration", 
                        help="Factor by which the pairwise affinities are exaggerated during the early iterations of optimization, default=4.",
                        type=int)
    # save plot of results
    parser.add_argument("-g", "--graph", 
                        help="Path where plot of t-SNE results, labeled by cluster, will be saved",
                       type=str)
    # PCA
    parser.add_argument('-P', "--PCA", action='store_true', help="Indicate whether or not PCA has been run on the input dataset, PCA can speed up the function's runtime")
    
    # parse args
    args = parser.parse_args()
    dataset = args.data
    output_file = args.output
    perplexity = args.perplexity
    learning_rate = args.learning_rate
    iterations = args.num_iterations
    early_exag = args.early_exaggeration
    graph = args.graph
    PCA = args.PCA

    # run diSNE with user inputs
    results = diSNE(dataset, perplexity, iterations, learning_rate, early_exag, PCA) 
    
    # save results to new file
    if not output_file: 
        diSNE_utils.save_data_to_h5ad(results)
    else:
        diSNE_utils.save_data_to_h5ad(results, output_file)
    
    # PLOTTING: only if user specified -g option
    # generate plot of results 
    if graph is not None: 
        # generate and save plot
        print("Generating plot")
        diSNE_utils.plot_results(results, graph, 'diSNE results', (10, 8))
    
    
if __name__ == "__main__":
    main()