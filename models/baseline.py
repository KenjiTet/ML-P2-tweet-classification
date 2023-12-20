from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split


import sys
import os

# Add the parent directory of 'utils' to the Python path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocessing import*  



SEED = 12345


def baseline(x_train, y_train, x_validation, y_validation):
	"""
    Trains a baseline text classification model using the Multinomial Naive Bayes algorithm and evaluates its accuracy.

    The function vectorizes the text data using CountVectorizer and then trains a MultinomialNB classifier.
    It then predicts labels for the validation set and prints the accuracy of the model on this set.

    Output:
    Prints the accuracy of the trained model on the validation dataset.
    """
	count_vect = CountVectorizer(max_features=80000,ngram_range=(1, 3))
	x_train_counts = count_vect.fit_transform(x_train)
	x_validation_counts = count_vect.transform(x_validation)
	clf = MultinomialNB().fit(x_train_counts, y_train)
	y_predicted = clf.predict(x_validation_counts)
	print(f'Baseline accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')




if __name__ == "__main__":
	
	df_tweet = pd.read_pickle(f"resources/tweet_full.pkl")
	df_tweet = df_tweet.sample(frac=1, random_state=1).reset_index(drop=True)
	
	X_train, X_test, y_train, y_test = train_test_split(df_tweet['tweet'], df_tweet['label'], test_size=0.2, random_state=42)
	baseline(X_train,y_train,X_test,y_test)
	