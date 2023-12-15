import pandas as pd
from nltk.tokenize import word_tokenize
import pandas as pd

from numpy import asarray
from numpy import zeros


import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def learn_w2v_embedding():

    df_full = pd.read_pickle("resources/tweet.pkl")
    tokenized_tweets = list(df_full["tweet"].apply(lambda x : word_tokenize(x))) 

    model = gensim.models.Word2Vec (tokenized_tweets, vector_size=200, window=20, min_count=4, workers=10)
    print("training process...")
    model.train(tokenized_tweets,total_examples=len(tokenized_tweets),epochs=10)

    word_vectors = model.wv 
    word_vectors.save_word2vec_format('resources/trained_w2v_embeddings.txt', binary=False) 

    print("Learned embedding saved")


