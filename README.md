# di-SNE
`di-SNE` is a tool to perform t-SNE, t-distributed stochastic neighbor embedding, on a pre-filtered scRNA-seq dataset. t-SNE aids with non-linear dimensionality reduction for high-dimensional data, making it applicable when analyzing single-cell RNA sequencing data, as well as many other types of datasets. `di-SNE` utilizes the existing Scanpy library to perform clustering on the given dataset, and uses the Matplotlib library to visualize the data. A full list of required packages can be found in `tests/requirements.txt`.   

# Usage  
`di-SNE`

## Options  
`perplexity`
`learning-rate`

