from utils.preprocessing import*
from embeddings.embedding_training import*
import argparse



if __name__ == '__main__':

    #python setup.py size small
    parser = argparse.ArgumentParser(description='Train or Evaluate a Model')
    parser.add_argument('--size', type=str, required=True, help='Mode to run: "full", "medium" or "small"')
    parser.add_argument('--dim', type=str, required=True, help='Mode to run: "full", "medium" or "small"')
    
    args = parser.parse_args()

    print("Preprocess the tweets...")
    preprocess_tweets(args.size)
    print("Finish preprocessing!")

    print("Learning embeddings...")
    learn_w2v_embedding(args.size, args.dim)
    print("Finish learning embeddings")
