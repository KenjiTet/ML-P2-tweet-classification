
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,MaxPooling1D,GRU
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt

from keras.models import load_model
import os



def save_model_if_better(model, model_type, score, current_best_score):
    """
    Save the model if it achieves better score than the current best.
    """
    model_folder = 'neural_networks/best_models_saved'
    model_path = os.path.join(model_folder, f'best_{model_type}.h5')

    if score > current_best_score:
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model.save(model_path)
        return True
    return False


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return its score.
    """
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])    
    return score[1]  


def evaluate_best_model(model_type, model, X_test, y_test):
    """
    Evaluate the model and return its score.
    """
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"=== Metrics of the best {model_type} ===")
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])    
    return score[1]  

def load_best_model(model_type):
    """
    Load and return the best saved model of a given type.
    """
    model_path = os.path.join('neural_networks/best_models_saved', f'best_{model_type}.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None
    

def prepare_data():
    df_tweet = pd.read_pickle("resources/tweet.pkl")
    df_tweet = df_tweet.sample(frac=1, random_state=1).reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df_tweet['tweet'], df_tweet['label'], test_size=0.05, random_state=42)

    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(X_train)

    vocab_size = len(tokenizer.word_index) + 1



    embedding_matrix = create_embedding_matrix(vocab_size, tokenizer)


    max_len = 100
    X_train, X_test = tok_and_pad(X_train,max_len,tokenizer), tok_and_pad(X_test,max_len,tokenizer)

    return X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len



def tok_and_pad(df,maxlen, tokenizer):
    """ 
    Tokenizes and pads to maxlen each tweet
    """

    df = tokenizer.texts_to_sequences(df)
    df = pad_sequences(df, padding='post', maxlen=maxlen) 

    return df

def create_embedding_matrix(vocab_size, tokenizer):
    """
    Creates the embedding matrix from the file that contains the pre-computed embedding vectors
    """
    
    #open file
    embeddings_dictionary = dict()
    glove_file = open('resources/trained_w2v_embeddings.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()
    
    #create matrix
    embedding_matrix = zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector    
    return embedding_matrix


def compute_predictions_nn(to_predict, threshold, model, tokenizer, maxlen):
    """
    Compute the predictions for a given dataset by using the trained model and threshold to map values to -1 or 1
    """
    to_predict = to_predict['tweet']
    to_predict = to_predict.astype(str)


    to_predict= tokenizer.texts_to_sequences(to_predict)

    to_predict = pad_sequences(to_predict, padding='post', maxlen=maxlen)

    result_test = model.predict(to_predict)
    print(result_test[:10])

    #it returns values between [0,1] (since sigmoid is used) 
    result_test[result_test < threshold] = -1 #replace values < threshold to -1
    result_test[result_test >= threshold] = 1
    
    return result_test



