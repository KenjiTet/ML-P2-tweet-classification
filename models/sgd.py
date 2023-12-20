import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from model_utils import*

import sys
import os

# Add the parent directory of 'utils' to the Python path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Now you can import from 'utils'
from preprocessing import*

SEED = 12345


def sgd_model(x_train, y_train, x_validation, y_validation, test_set):
	"""
    Trains a Stochastic Gradient Descent (SGD) Classifier on TF-IDF transformed text data, evaluates its accuracy on a 
    validation set, and makes predictions on a test set.

    The function first transforms the input text data into TF-IDF vectors, then trains an SGD Classifier with 
    'modified_huber' loss, evaluates its performance on the validation set, and finally uses the trained model to make 
    predictions on the test set.

    Returns:
    y_sub (array-like): Predicted labels for the test set.
    """	
	x_train, x_validation, X_pred = prepare_tfidf(x_train, x_validation, test_set)
	clf = SGDClassifier(tol=1e-3, loss='modified_huber').fit(x_train, y_train)
	y_predicted = clf.predict(x_validation)
	print(f'SGD classifier accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')
	y_sub = clf.predict(X_pred)
	return y_sub


if __name__ == "__main__":
	
	
	df_tweet = pd.read_pickle(f"resources/tweet_full.pkl")
	df_tweet = df_tweet.sample(frac=1, random_state=1).reset_index(drop=True)
	test_set = preprocess_tweets_to_predict()

	X_train, X_test, y_train, y_test = train_test_split(df_tweet['tweet'], df_tweet['label'], test_size=0.2, random_state=42)
	
	y_submission = sgd_model(X_train, y_train, X_test, y_test, test_set)
	y_submission[y_submission == 0] = -1
	ids=np.arange(1,len(y_submission)+1)

	create_csv_submission(ids, y_submission, "predictions/pred_basic_sgd_full.csv")