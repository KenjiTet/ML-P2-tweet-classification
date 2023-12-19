
# coding: utf-8

#Keras import
from keras.models import Sequential
from keras.layers import  Dense
from keras.layers import Flatten
from keras.layers import Embedding
from .nn_utils import*

#Keras import
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Embedding
from keras.optimizers import Adam


BATCH_SIZE = 1024
EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1





def simple_nn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, lr, batch_size = BATCH_SIZE):
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

    # Load the best NN model's score
    best_nn_model = load_best_model('simple_nn', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'simple_nn', current_score, best_score):
        print("New best NN model saved.")
    else:
        print("Current NN model did not outperform the best one.")


    return model




def cnn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, dropout_rate, filters_list, kernel_sizes, batch_size=BATCH_SIZE):
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

    # Load the best NN model's score
    best_nn_model = load_best_model('cnn', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(size, model, 'cnn', current_score, best_score):
        print("New best CNN model saved.")
    else:
        print("Current CNN model did not outperform the best one.")

    return model


