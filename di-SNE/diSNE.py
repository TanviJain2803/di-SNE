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
    
   if __name__ == "__main__":
       main()
    
