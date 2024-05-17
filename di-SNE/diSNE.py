import argparse

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
    parser.add_argument("-p","--perplexity", help="perplexity value used in tSNE, recommended range: _", type=int) #float?
    # number of iterations
    parser.add_argument("-n","--num-iterations", help="number of iterations used in tSNE, default: _", type=int)
    # learning rate
    parser.add_argument("-r","--learning-rate", help="learning rate used in tSNE, default: _", type=float) #int?
    # parse args
    args = parser.parse_args()

#dataset = args.data
#perplexity = args.perplexity
#iterations = args.num_iterations
#learning_rate = args.learning_rate
    
   if __name__ == "__main__":
       main()
    
