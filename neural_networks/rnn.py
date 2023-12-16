
#Keras import

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Masking
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM,Bidirectional
from keras.layers import Embedding
from .nn_utils import*


BATCH_SIZE = 1024
DIM = 200
EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1



def rnn_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen):
    """
    Create the recurrent neural network model for a long-short term memory network
    """

    #Create model
    model = Sequential()
    model.add(Embedding(vocab_size, DIM, weights=[embedding_matrix], input_length=maxlen,trainable=False, mask_zero=True))
    model.add(Masking(mask_value=0.0)) #need masking layer to not train on padding
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    #Fit model
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose =  VERBOSE, validation_split = VALIDATION_SPLIT)

    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose = VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('rnn_lstm')
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(model, 'rnn_lstm', current_score, best_score):
        print("New best rnn_lstm model saved.")
    else:
        print("Current rnn_lstm model did not outperform the best one.")


    return model

def rnn_bi_lstm(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a Bi-Directional Long-Short Term Memory network
    """
    #Create model
    model = Sequential()
    model.add(Embedding(vocab_size, DIM, weights=[embedding_matrix], input_length=maxlen,trainable=False, mask_zero=True))
    model.add(Masking(mask_value=0.0)) #need masking layer to not train on padding
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    #Fit model
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose =  VERBOSE, validation_split = VALIDATION_SPLIT)
    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose = VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('rnn_bi_lstm')
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(model, 'rnn_bi_lstm', current_score, best_score):
        print("New best rnn_bi_lstm model saved.")
    else:
        print("Current rnn_bi_lstm model did not outperform the best one.")


    return model




def rnn_gru(X_train,y_train,X_test,y_test,vocab_size,embedding_matrix,maxlen):
    """
    Create the model for a Convolutional Neural Network with Gated Recurrent Unit
    """
    #Create model
    model = Sequential()
    embedding_layer = Embedding(vocab_size, DIM, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model.add(embedding_layer)
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    #Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    #Fit model
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose =  VERBOSE, validation_split = VALIDATION_SPLIT)
    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose = VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('rnn_gru')
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(model, 'rnn_gru', current_score, best_score):
        print("New best rnn_gru model saved.")
    else:
        print("Current rnn_gru model did not outperform the best one.")


    return model