
# coding: utf-8

#Keras import
from keras.models import Sequential
from keras.layers import  Dense
from keras.layers import Flatten
from keras.layers import Embedding
from .nn_utils import*

BATCH_SIZE = 1024
DIM = 20
EPOCHS = 6
VALIDATION_SPLIT = 0.05
VERBOSE = 1


def simple_nn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len):
    """
    Create the model for a Simple Neural Net
    OUTPUT:
    Returns the model trained 
    """
    #Create model
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_len , trainable=False) #trainable set to False bc we use the downloaded dict
    model.add(embedding_layer)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())
    #Fit model
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose =  VERBOSE, validation_split = VALIDATION_SPLIT)

    #Evaluate model
    score = model.evaluate(X_test, y_test, verbose = VERBOSE)



    current_score = score[1]

    # Load the best NN model's score
    best_nn_model = load_best_model('nn')
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it is better
    if save_model_if_better(model, 'simple_nn', current_score, best_score):
        print("New best NN model saved.")
    else:
        print("Current NN model did not outperform the best one.")


    return model


