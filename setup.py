from utils.preprocessing import*
from embeddings.embedding_training import*
  


if __name__ == '__main__':

    print("Preprocess the tweets...")
    preprocess_tweets()
    print("Finish preprocessing!")

    print("Learning embeddings...")
    learn_w2v_embedding()
    print("Finish learning embeddings")
