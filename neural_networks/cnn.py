
from .nn_utils import*
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


def simple_nn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, lr, batch_size=BATCH_SIZE):
    """
    Trains a simple neural network for binary classification using pre-trained embeddings.

    Parameters:
    - size (int): A parameter to specify the dataset size or model complexity.
    - X_train (array-like): Training data.
    - y_train (array-like): Training labels.
    - X_test (array-like): Test data.
    - y_test (array-like): Test labels.
    - vocab_size (int): Size of the vocabulary.
    - embedding_matrix (array-like): Pre-trained word embeddings.
    - max_len (int): Maximum length of the sequences.
    - dim (int): Dimension of the embedding vectors.
    - lr (float): Learning rate for the optimizer.
    - batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.

    Returns:
    - model (Keras Model): The trained Keras model.
    """

    # Initialize the optimizer
    optimizer = Adam(learning_rate=lr)

    # Build the model
    model = Sequential()
    embedding_layer = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)  # Using pre-trained embeddings
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Print model summary
    print(model.summary())

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    # Evaluate the model on test data
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load and evaluate the best saved model
    best_nn_model = load_best_model('simple_nn', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it performs better
    if save_model_if_better(size, model, 'simple_nn', current_score, best_score):
        print("New best NN model saved.")
    else:
        print("Current NN model did not outperform the best one.")

    return model



def cnn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, maxlen, dim, lr, dropout_rate, filters_list, kernel_sizes, batch_size=BATCH_SIZE):
    """
    Trains a Convolutional Neural Network (CNN) for binary classification using pre-trained embeddings.

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
    - dropout_rate (float): Dropout rate for regularization.
    - filters_list (list of ints): List of filter numbers for each Conv1D layer.
    - kernel_sizes (list of ints): List of kernel sizes for each Conv1D layer.
    - batch_size (int, optional): Batch size for training. Defaults to BATCH_SIZE.

    Returns:
    - model (Keras Model): The trained Keras model.
    """

    # Initialize the optimizer
    optimizer = Adam(learning_rate=lr)

    # Build the CNN model
    model = Sequential()
    model.add(Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=maxlen))
    model.add(Dropout(dropout_rate))

    # Add Conv1D layers
    for filters, kernel_size in zip(filters_list, kernel_sizes):
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    
    model.add(Flatten())
    model.add(Dense(max(filters_list)))  
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Print model summary
    print(model.summary())

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    # Evaluate the model on test data
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    current_score = score[1]

    # Load and evaluate the best saved model
    best_nn_model = load_best_model('cnn', size)
    best_score = evaluate_model(best_nn_model, X_test, y_test) if best_nn_model else 0

    # Save the current model if it performs better
    if save_model_if_better(size, model, 'cnn', current_score, best_score):
        print("New best CNN model saved.")
    else:
        print("Current CNN model did not outperform the best one.")

    return model