import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from model_utils import*
import sys
import os

# Add the parent directory of 'utils' to the Python path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



# Now you can import from 'utils'
from utils.preprocessing import*

SEED = 12345

def svc_model(x_train,y_train,x_validation,y_validation, test_set):
	"""Trains an Support Vector Machine Classifier on tfidf data and prints accuracy"""
	x_train_tfidf,x_validation_tfidf, X_pred = prepare_tfidf(x_train,y_train,x_validation,y_validation, test_set)
	clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=3).fit(x_train_tfidf, y_train)
	y_predicted = clf.predict(x_validation_tfidf)
	print(f'SVC accuracy on validation set is {metrics.accuracy_score(y_validation, y_predicted)}')
	y_subm = clf.predict(X_pred)
	return y_subm


if __name__ == "__main__":
	

	#Clean training set and test set
	train_set, test_set = train_test_cleaner()

	#Split into training set and validation set
	x = train_set.tweet
	y = train_set.label
	x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=.3, random_state=SEED)
	
	
	y_submission = svc_model(x_train,y_train,x_validation,y_validation, test_set)
	ids=np.arange(1,len(y_submission)+1)

	create_csv_submission(ids, y_submission, "models/Submissions/submission_svc_model.csv")
