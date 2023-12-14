from sklearn.model_selection import train_test_split
from cnn import*
from nn_utils import tok_and_pad_for_nn, compute_predictions_nn
import time
import pickle

import sys
import os

# Add the parent directory of 'utils' to the Python path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import from 'utils'
from utils.preprocessing import*
from utils.embedding import create_embedding_matrix
from utils.preprocessing import train_test_cleaner
from utils.loads import load_vocabulary, load_glove_embeddings

"""
print("load embedding and vocab")
embeddings = load_glove_embeddings('resources/embeddings.npy')
vocab = load_vocabulary('resources/vocab.pkl')


MAXLEN = 40
NUM_WORD = len(vocab)
EMBED_DIM = embeddings.shape[1] # = 20
"""


train_set, unknown = train_test_df()

with open("tweet.pkl", "wb") as f:
    pickle.dump(train_set, f)

"""
train_set.sample(frac=1, random_state=1).reset_index(drop=True)


#split training set
X_train, X_test, y_train, y_test = train_test_split(train_set['tweet'], train_set['label'], test_size=0.1, random_state=42)


print("tok")
start_time = time.time()  # Record the start time
#prepare input for neural nets
X_train , X_test, vocab_size, tokenizer  = tok_and_pad_for_nn(X_train, X_test, MAXLEN , vocab)


end_time = time.time()  # Record the end time
total_time = end_time - start_time  # Calculate the total time taken

print(f"TOK: {total_time} seconds .")


#build embedding matrix
embedding_matrix = create_embedding_matrix(vocab_size-1, tokenizer)



#Train and run first model + prediction on unknown set
#4-convolutional neural net
print("train")
start_time = time.time()  # Record the start time
#prepare input for neural nets
model_1 = simple_nn(X_train , y_train, X_test, y_test, NUM_WORD, embedding_matrix , MAXLEN)

end_time = time.time()  # Record the end time
total_time = end_time - start_time  # Calculate the total time taken
print(f"TRAIN: {total_time} seconds .")


print("pred")
start_time = time.time()  # Record the start time
#prepare input for neural nets
preds_1 = compute_predictions_nn(to_predict = unknown, threshold = 0.51, model = model_1, tokenizer = tokenizer, maxlen=MAXLEN)

end_time = time.time()  # Record the end time
total_time = end_time - start_time  # Calculate the total time taken
print(f"PRED: {total_time} seconds .")
"""