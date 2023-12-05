import numpy as np
import pickle

def load_tweets(file_path):
    """
    Load tweets from a text file.

    Args:
    file_path (str): Path to the text file containing tweets.

    Returns:
    list: A list of tweets.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

def load_glove_embeddings(embedding_path):
    """
    Load the GloVe word embeddings from the specified path.
    """
    return np.load(embedding_path)

def load_vocabulary(vocab_path):
    """
    Load the vocabulary from the specified path.
    """
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab