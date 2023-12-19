
#Keras import

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Masking
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM,Bidirectional
from keras.layers import Embedding
from .nn_utils import*
from keras.optimizers import Adam


BATCH_SIZE = 1024
EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1




def rnn_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, lstm_layers, dropout_rate, recurrent_dropout_rate, batch_size=BATCH_SIZE):
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen, trainable=False, mask_zero=True))
    model.add(Masking(mask_value=0.0))

    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1  # Only the last layer should not return sequences
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
    best_nn_model = load_best_model('rnn_lstm', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'rnn_lstm', current_score, best_score):
        print("New best rnn_lstm model saved.")
    else:
        print("Current rnn_lstm model did not outperform the best one.")


    return model


def rnn_bi_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, bi_lstm_layers, dropout_rate, recurrent_dropout_rate, batch_size= BATCH_SIZE):
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
    best_nn_model = load_best_model('rnn_bi_lstm', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'rnn_bi_lstm', current_score, best_score):
        print("New best rnn_bi_lstm model saved.")
    else:
        print("Current rnn_bi_lstm model did not outperform the best one.")


    return model




def rnn_gru(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, hidden_units, gru_layers, dropout_rate, batch_size= BATCH_SIZE):
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))

    for i in range(gru_layers):
        return_sequences = i < gru_layers - 1  # Only the last layer should not return sequences
        model.add(GRU(hidden_units, return_sequences=return_sequences, dropout=dropout_rate))

    model.add(Flatten())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('rnn_gru', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'rnn_gru', current_score, best_score):
        print("New best rnn_gru model saved.")
    else:
        print("Current rnn_gru model did not outperform the best one.")


    return model