from sklearn.feature_extraction.text import TfidfVectorizer
import csv



def prepare_tfidf(x_train, x_validation, test_set):
    """
    Transforms text data into TF-IDF vectors for the training, validation, and test datasets.
    The function uses TfidfVectorizer to convert the text data into TF-IDF vectors.

    Returns:
    Tuple containing TF-IDF vectors for the training data, validation data and test data.
    """
    transformer = TfidfVectorizer(max_features=80000, ngram_range=(1, 3))
    x_train = transformer.fit_transform(x_train)
    x_validation = transformer.transform(x_validation)
    X_pred = transformer.transform(test_set['tweet'])  
    
    return x_train, x_validation, X_pred



def create_csv_submission(ids, y_pred, name):
    """
    Creates a CSV file for submission with predictions.

    Parameters:
    ids (list or array-like): The list of IDs corresponding to the predictions.
    y_pred (list or array-like): Predicted labels or values.
    name (str): Name of the CSV file to be created.

    The function writes a CSV file with two columns: 'Id' and 'Prediction', 
    where 'Id' is taken from the 'ids' parameter and 'Prediction' from 'y_pred'.
    """
    
    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

