The Jupyter notebooks in this repository contain example usage of the diSNE functions on small test datasets for debugging and testing. 

This repository also contains a toy dataset to test/showcase the basic functionality of diSNE. The dataset is called `example_dataset.h5ad`, and the following command can be used to run diSNE on the file and generate a plot of the results:  
`diSNE -g example_plot.png example_dataset.h5ad`

This command will generate a new `h5ad` file titled `diSNE_output.h5ad` and a graph of the results titled `example_plot.png` in your current working directory. You can try adjusting the perplexity, number of iterations, and early exaggeration values, although the default options should be effective on this example dataset. You should see three distinct clusters in the plot generated, although the exact plot may look slightly different from the reference plot we provided, since t-SNE is a stochastic/randomized algorithm, and results vary across runs.
