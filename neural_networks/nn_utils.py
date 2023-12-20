import os
import pandas as pd

from numpy import asarray
from numpy import zeros

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


def save_model_if_better(size, model, model_type, score, current_best_score):
    """
    Saves the trained model if it achieves a better score than the currently best saved model.

    Parameters:
    - size (int/str): A parameter to specify the dataset size or model complexity.
    - model (Keras Model): The trained Keras model to be saved.
    - model_type (str): A string indicating the type of the model (e.g., 'cnn', 'simple_nn').
    - score (float): The performance score of the 'model' on the test/validation set.
    - current_best_score (float): The best score achieved by the previously saved model of the same type.

    Returns:
    - True if the model is saved, False otherwise.
    """

    model_folder = 'neural_networks/best_models_saved'
    model_path = os.path.join(model_folder, f'best_{model_type}_{size}.h5')

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
    if model is None:
        print("No model to evaluate")
        return 0
    else:
        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])    
        return score[1]  


def evaluate_best_model(model_type, model, X_test, y_test):
    """
    Evaluate the best model and return its score.
    """
    if model is None:
        print("No model to evaluate")
        return 0
    else:
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f"=== Metrics of the best {model_type} ===")
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])    
        return score[1]  


def load_best_model(model_type, size):
    """
    Load and return the best saved model of a given type.
    """
    model_path = os.path.join('neural_networks/best_models_saved', f'best_{model_type}_{size}.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        print(f"No such model exist: python run.py --mode train model_type {model_type} --size {size}")
        return None
    


def prepare_data(size, embedding_dim, max_len):
    """
    Prepares tweet data for machine learning models, including tokenization and embedding matrix creation.

    Parameters:
    - size (int/str): A parameter to specify the dataset size.
    - embedding_dim (int): Dimension of the embedding vectors.
    - max_len (int): Maximum length of the sequences after tokenization.

    Returns:
    - X_train, X_test: Tokenized and padded training and test data.
    - y_train, y_test: Corresponding labels for training and test data.
    - vocab_size: Size of the vocabulary.
    - tokenizer: The tokenizer object used for tokenizing tweets.
    - embedding_matrix: Matrix containing embeddings for each word in the vocabulary.
    - max_len: Maximum length of the sequences.
    - embedding_dim: Dimension of the embedding vectors.
    """

    # Load and shuffle the dataset
    df_tweet = pd.read_pickle(f"resources/tweet_{size}.pkl")
    df_tweet = df_tweet.sample(frac=1, random_state=1).reset_index(drop=True)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_tweet['tweet'], df_tweet['label'], test_size=0.05, random_state=42)

    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(X_train)

    # Calculate the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # Create the embedding matrix
    embedding_matrix = create_embedding_matrix(vocab_size, tokenizer, embedding_dim, size)

    # Tokenize and pad the tweets
    X_train = tok_and_pad(X_train, max_len, tokenizer)
    X_test = tok_and_pad(X_test, max_len, tokenizer)

    return X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, embedding_dim





def tok_and_pad(df, maxlen, tokenizer):
    """ 
    Tokenizes and pads to maxlen each tweet
    """
    df = tokenizer.texts_to_sequences(df)
    df = pad_sequences(df, padding='post', maxlen=maxlen) 
    return df


def create_embedding_matrix(vocab_size, tokenizer, embedding_dim, size):
    """
    Creates an embedding matrix using pre-computed word embeddings.

    Parameters:
    - vocab_size (int): The size of the vocabulary.
    - tokenizer (Tokenizer): The tokenizer object used for tokenizing the tweets.
    - embedding_dim (int): The dimension of the embedding vectors.
    - size (int): A parameter to specify the dataset size or model complexity.

    Returns:
    - embedding_matrix (array-like): A matrix where each row index corresponds to a word in the vocabulary and contains 
      the embedding vector for that word.
    """

    # Load pre-computed word embeddings from a file
    embeddings_dictionary = dict()
    glove_file = open(f'resources/trained_w2v_embeddings_{size}_{embedding_dim}.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    
    # Create the embedding matrix
    embedding_matrix = zeros((vocab_size, embedding_dim))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix




def compute_predictions_nn(to_predict, model, tokenizer, maxlen):
    """
    Computes predictions for a given dataset using a trained neural network model.

    Parameters:
    - to_predict (DataFrame): The dataset containing the text data to predict. Assumes a column named 'tweet'.
    - model (Keras Model): The trained Keras neural network model.
    - tokenizer (Tokenizer): The tokenizer used for tokenizing the text data.
    - maxlen (int): The maximum length of sequences after padding.

    Returns:
    - result_test (array-like): The array of predictions, with values -1 or 1.
    """

    # Define the threshold for mapping predictions
    threshold = 0.5

    # Extract and preprocess the tweet data
    to_predict = to_predict['tweet']
    to_predict = to_predict.astype(str)
    to_predict = tokenizer.texts_to_sequences(to_predict)
    to_predict = pad_sequences(to_predict, padding='post', maxlen=maxlen)

    # Predict using the model
    result_test = model.predict(to_predict)

    # Map predictions to -1 or 1 based on the threshold
    result_test[result_test < threshold] = -1  
    result_test[result_test >= threshold] = 1  

    return result_test



