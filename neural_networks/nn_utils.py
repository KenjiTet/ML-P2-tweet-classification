
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def tok_and_pad_for_nn(X_train, X_test, max_len, vocab):
    """ 
    Tokenizes and pads to maxlen each tweet for each given dataframe
    so that it can directly be used in a neural network
    INPUT: 
        X_train: Panda Series       - tweets
        X_test: Panda Series        - tweets
        maxlen: int                 - maximal length of word we take per tweet
        vocab : dict              - mapping from vocab to index
    """
    
    # Create a tokenizer and fit on the training data
    tokenizer = Tokenizer(num_words=len(vocab))  # vocab_dict is your word index dictionary
    tokenizer.word_index = vocab  # Set the tokenizer's word index to your vocab_dict
    tokenizer.index_word = {i: word for word, i in vocab.items()}  # Create reverse mapping

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    return X_train_padded, X_test_padded,vocab_size, tokenizer


def tok_and_pad_for_unknown(unknown):
    return 0


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



