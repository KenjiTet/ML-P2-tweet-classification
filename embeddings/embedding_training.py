import pandas as pd
from nltk.tokenize import word_tokenize
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def learn_w2v_embedding(size, dim):
    """
    Learns Word2Vec embeddings from a dataset of tweets.

    Parameters:
    size : Used to specify the dataset size by referencing a file with the corresponding name.
    dim : The dimensionality of the Word2Vec embedding vectors.

    The function reads a dataset of tweets, tokenizes them, and then uses gensim's Word2Vec model 
    to learn embeddings. The learned embeddings are saved in a text file.
    """

    # Convert dim to integer
    dim = int(dim)

    # Load the dataset of tweets
    df_full = pd.read_pickle(f"resources/tweet_{size}.pkl")

    # Tokenize the tweets
    tokenized_tweets = list(df_full["tweet"].apply(lambda x: word_tokenize(x)))

    # Initialize and train the Word2Vec model
    model = gensim.models.Word2Vec(tokenized_tweets, vector_size=dim, window=20, min_count=4, workers=10)
    print("training process...")
    model.train(tokenized_tweets, total_examples=len(tokenized_tweets), epochs=10)

    # Save the learned word vectors
    word_vectors = model.wv
    word_vectors.save_word2vec_format(f'resources/trained_w2v_embeddings_{size}_{dim}.txt', binary=False)

    print("Learned embedding saved")