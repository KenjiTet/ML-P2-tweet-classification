from .nn_utils import*

from keras.models import Sequential
from keras.layers import Dropout, Dense, Masking
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM,Bidirectional
from keras.layers import Embedding
from keras.optimizers import Adam

BATCH_SIZE = 1024
EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1

def rnn_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, lstm_layers, dropout_rate, recurrent_dropout_rate, batch_size=BATCH_SIZE):
    """
    Builds and trains an RNN with LSTM layers for binary classification.

    Parameters:
    - size (int): A parameter to specify the dataset size or model complexity.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.
    - X_test (array-like): Test data.
    - y_test (array-like): Test labels.
    - vocab_size (int): Size of the vocabulary.
    - embedding_matrix (array-like): Pre-trained word embeddings.
    - maxlen (int): Maximum length of the sequences.
    - dim (int): Dimension of the embedding vectors.
    - lr (float): Learning rate for the optimizer.
    - hidden_units (int): Number of units in the LSTM layers.
    - lstm_layers (int): Number of LSTM layers in the model.
    - dropout_rate (float): Dropout rate for regularization.
    - recurrent_dropout_rate (float): Recurrent dropout rate for the LSTM layers.
    - batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.

    Returns:
    - model (Keras Model): The trained Keras model.
    """

    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen, trainable=False, mask_zero=True))
    model.add(Masking(mask_value=0.0))

    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1  
        model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))

    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('lstm', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'lstm', current_score, best_score):
        print("New best rnn_lstm model saved.")
    else:
        print("Current rnn_lstm model did not outperform the best one.")


    return model


def rnn_bi_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, bi_lstm_layers, dropout_rate, recurrent_dropout_rate, batch_size= BATCH_SIZE):
    """
    Builds and trains an RNN with Biderectional LSTM layers for binary classification.

    Parameters:
    - size (int): A parameter to specify the dataset size or model complexity.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.
    - X_test (array-like): Test data.
    - y_test (array-like): Test labels.
    - vocab_size (int): Size of the vocabulary.
    - embedding_matrix (array-like): Pre-trained word embeddings.
    - maxlen (int): Maximum length of the sequences.
    - dim (int): Dimension of the embedding vectors.
    - lr (float): Learning rate for the optimizer.
    - hidden_units (int): Number of units in the LSTM layers.
    - lstm_layers (int): Number of LSTM layers in the model.
    - dropout_rate (float): Dropout rate for regularization.
    - recurrent_dropout_rate (float): Recurrent dropout rate for the LSTM layers.
    - batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.

    Returns:
    - model (Keras Model): The trained Keras model.
    """
    
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen, trainable=False, mask_zero=True))
    model.add(Masking(mask_value=0.0))

    for i in range(bi_lstm_layers):
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=i < bi_lstm_layers - 1, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)))

    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('bi_lstm', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'bi_lstm', current_score, best_score):
        print("New best rnn_bi_lstm model saved.")
    else:
        print("Current rnn_bi_lstm model did not outperform the best one.")


    return model




def rnn_gru(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, gru_layers, dropout_rate, batch_size= BATCH_SIZE):
    """
    Builds and trains an RNN with GRU layers for binary classification.

    Parameters:
    - size (int/str): A parameter to specify the dataset size or model complexity.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.
    - X_test (array-like): Test data.
    - y_test (array-like): Test labels.
    - vocab_size (int): Size of the vocabulary.
    - embedding_matrix (array-like): Pre-trained word embeddings.
    - maxlen (int): Maximum length of the sequences.
    - dim (int): Dimension of the embedding vectors.
    - lr (float): Learning rate for the optimizer.
    - hidden_units (int): Number of units in the GRU layers.
    - gru_layers (int): Number of GRU layers in the model.
    - dropout_rate (float): Dropout rate for regularization.
    - batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.

    Returns:
    - model (Keras Model): The trained Keras model.
    """
    
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))

    for i in range(gru_layers):
        model.add(GRU(hidden_units, return_sequences=True))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('gru', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'gru', current_score, best_score):
        print("New best rnn_gru model saved.")
    else:
        print("Current rnn_gru model did not outperform the best one.")

    return model


