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
#     parser.add_argument()
    
    # parse args
    args = parser.parse_args()
    
   if __name__ == "__main__":
       main()
    