from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


import sys
import os

# Add the parent directory of 'utils' to the Python path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now you can import from 'utils'
from utils.preprocessing import*  # Replace 'some_function' with the actual function/class name you want to import

# rest of your baseline.py code


SEED = 12345


def baseline(x_train,y_train,x_validation,y_validation):
	"""Trains a naive bayes model on count vectorized data and prints accuracy"""
	count_vect = CountVectorizer(max_features=80000,ngram_range=(1, 3))
	x_train_counts = count_vect.fit_transform(x_train)
	x_validation_counts = count_vect.transform(x_validation)
	clf = MultinomialNB().fit(x_train_counts, y_train)
	y_predicted = clf.predict(x_validation_counts)
	print(f'Baseline accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')




if __name__ == "__main__":
	
	
	#Clean training set and test set
	train_set, test_set = train_test_cleaner()

	#Split into training set and validation set
	x = train_set.tweet
	y = train_set.label
	x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=.3, random_state=SEED)
	
	#Train the model selected and print accuracy on validation set
	baseline(x_train,y_train,x_validation,y_validation)
	