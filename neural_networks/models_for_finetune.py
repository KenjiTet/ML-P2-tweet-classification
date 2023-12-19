from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Masking
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM,Bidirectional
from keras.layers import Embedding
from nn_utils import*
from keras.optimizers import Adam
from keras.layers import Conv1D


EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1



def finetune_simple_nn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr):
    """
    Create the model for a Simple Neural Net
    OUTPUT:
    Returns the model trained 
    """
    #Create model
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    embedding_layer = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len , trainable=False) #trainable set to False bc we use the downloaded dict
    model.add(embedding_layer)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())
    #Fit model
    model.fit(X_train, y_train, batch_size = batch_size, epochs = EPOCHS, verbose =  VERBOSE, validation_split = VALIDATION_SPLIT)

    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose = VERBOSE)
    current_score = score[1]


    return model, current_score





def finetune_cnn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, batch_size, lr, dropout_rate, filters_list, kernel_sizes):
    optimizer = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen))
    model.add(Dropout(dropout_rate))

    for filters, kernel_size in zip(filters_list, kernel_sizes):
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    
    model.add(Flatten())
    model.add(Dense(max(filters_list)))  # Example: using the max number of filters as the size of the dense layer
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    return model, current_score


def finetune_rnn_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, batch_size, lr, hidden_units, lstm_layers, dropout_rate, recurrent_dropout_rate):
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

    return model, current_score



def finetune_rnn_bi_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, batch_size, lr, hidden_units, bi_lstm_layers, dropout_rate, recurrent_dropout_rate):
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

    return model, current_score




def finetune_rnn_gru(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, batch_size, lr, hidden_units, gru_layers, dropout_rate):
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

    return model, current_score


