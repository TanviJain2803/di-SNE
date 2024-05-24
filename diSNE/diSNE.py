#!/usr/bin/env python

import sys
import os
sys.path.append(os.environ["HOME"]+"/.local/lib/python3.9/site-packages")
# set up packages 
import argparse
import scanpy as sc, anndata as ad
import numpy as np
import leidenalg
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform
# import functions from utils file
from .diSNE_utils import *


def diSNE(adata, perplexity, T, learning_rate: int = 200, early_exaggeration: int = 4, n_dim: int = 2):
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm implementation.
    Using the results from the helper functions defined in diSNE_utils.py, this function iterates 
    and optimizes low-dimensional mapping, using a gradient descent with momentum.

    Args:
        adata (AnnData object): Input AnnData object representing the user's dataset.
        perplexity (int, optional): Perplexity parameter. Default is 10.
        T (int, optional): Number of iterations for optimization. Default is 1000.
        learning_rate (int, optional): Learning rate for updating the low-dimensional embeddings, or controlling
        the step size at each iteration. Default is 200.
        early_exaggeration (int, optional): Factor by which the pairwise affinities are exaggerated
            during the early iterations of optimization. Default is 4.
        n_dim (int, optional): The number of dimensions of the low-dimensional embeddings. Default is 2.

    Returns:
        adata (AnnData object): Updated AnnData object containing the t-SNE results.

    """
    X = adata.X
    n = len(X)

    # Get original affinities matrix
    P = get_highdim_affinities(X, perplexity)
    P_symmetric = convert_to_jointprob(P)

    # Initialization
    Y = np.zeros(shape=(T, n, n_dim))
    Y_minus1 = np.zeros(shape=(n, n_dim))
    Y[0] = Y_minus1
    Y1 = initialize(X, n_dim)
    Y[1] = np.array(Y1)

    print("Optimizing Low Dimensional Embedding....")
    # Optimization of the low-dimensional embedding
    for t in range(1, T - 1):
        # Momentum & Early Exaggeration
        if t < 250:
            momentum = 0.5
            early_exaggeration = early_exaggeration
        else:
            momentum = 0.8
            early_exaggeration = 1

        # Get Low Dimensional Affinities
        Q = get_lowdim_affinities(Y[t])

        # Get Gradient of Cost Function
        gradient = compute_gradient(early_exaggeration * P_symmetric, Q, Y[t])

        # Update Rule
        Y[t + 1] = Y[t] - learning_rate * gradient + momentum * (Y[t] - Y[t - 1])  # Use negative gradient

        # Compute current value of cost function (Kullback-Leibler divergence)
        if t % 100 == 0:
            cost = np.sum(P_symmetric * np.log(P_symmetric / Q))
            print(f"Iteration {t}: Value of Cost Function is {cost}")

    print(
        f"Completed Low Dimensional Embedding: Final Value of Cost Function is {np.sum(P_symmetric * np.log(P_symmetric / Q))}"
    )
    soln = Y[-1]
#     return soln, Y
    adata.obsm['X_tsne'] = soln
    return

def main():
    parser = argparse.ArgumentParser(
        prog="diSNE",
        description="Command-line tool to perform t-SNE (t-Distributed Stochastic Neighbor Embedding) on a pre-filtered, clustered dataset"
    )
    
    # input 
    parser.add_argument("data", help="Annotated data (AnnData) matrix, with Leiden clustering already performed", type=str)
    
    # output 
    # option to display output?
    
    # other options
    # perplexity
    parser.add_argument("-p", "--perplexity", 
                        help="Perplexity value used in the t-SNE algorithm. It is recommended to use a larger perplexity for larger datasets. Default=10, recommended range: 5-50", 
                        type=float) 
    # learning rate
    parser.add_argument("-r", "--learning-rate", 
                        help="Learning rate used during optimization, default=200. Recommended range: 100-1000",
                        type=float)
    # number of iterations 
    parser.add_argument("-T", "--num-iterations", 
                        help="Number of iterations used for optimization, default=1000",
                        type=float) 
    # early exaggeration
    parser.add_argument("-E", "--early-exaggeration", 
                        help="Factor by which the pairwise affinities are exaggerated during the early iterations of optimization, default=4.",
                        type=float)
    
    # parse args
    args = parser.parse_args()

    dataset = args.data
    perplexity = args.perplexity
    learning_rate = args.learning_rate
    iterations = args.num_iterations
    early_exag = args.early_exaggeration

    # check arg parser functionality
    print("dataset:", dataset)
    print("perp:", perplexity)
    print("learning rate:", learning_rate)
    print("iterations:", iterations)
    print("dataset:", early_exag)

    # run diSNE with user inputs
#     diSNE(dataset) 
    
if __name__ == "__main__":
    main()
    
