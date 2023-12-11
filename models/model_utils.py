from sklearn.feature_extraction.text import TfidfVectorizer
import csv

def prepare_tfidf(x_train, y_train, x_validation, y_validation, test_set):
    tfidf_transformer = TfidfVectorizer(max_features=80000, ngram_range=(1, 3))
    x_train_tfidf = tfidf_transformer.fit_transform(x_train)
    x_validation_tfidf = tfidf_transformer.transform(x_validation)
    X_pred = tfidf_transformer.transform(test_set['tweet'])  # Assuming 'test_set' is a DataFrame and 'Tweets' is the column name
    
    return x_train_tfidf, x_validation_tfidf, X_pred



def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

