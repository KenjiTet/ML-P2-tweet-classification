from utils.preprocessing import*
from neural_networks.nn_utils import*
from embeddings.embedding_training import*
import argparse


import csv
import glob
import numpy as np


NUM_PREDICTION_ROWS = 10000


def majority_voting():

    pred_files = glob.glob('predictions/*.csv')
    predictions = np.zeros((NUM_PREDICTION_ROWS, 2))
    
    for file in pred_files:
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
            current_preds = np.array([int(l.split(',')[1]) for l in lines])
            current_preds[current_preds < 0] = 0
            predictions[range(NUM_PREDICTION_ROWS), current_preds] += 1

    predictions = np.argmax(predictions, axis=1)
    predictions[predictions < 1 ] = -1

    create_csv_submission(predictions, 'predictions/majority_vote_preds.csv')

    return 0



def create_csv_submission(y_pred, path):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               path (string name of .csv output file to be created)
    """
    ids=[i for i in range(1,len(y_pred)+1)]
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



if __name__ == '__main__':

    #python setup.py size small
    parser = argparse.ArgumentParser(description='Precise the size of the dataset on which the models where trained')
    parser.add_argument('--size', type=str, required=True, help='Mode to run: "full", "medium" or "small"')
    
    args = parser.parse_args()

    best_nn = load_best_model("simple_nn", args.size)
    best_cnn = load_best_model("cnn", args.size)
    best_lstm = load_best_model("rnn_lstm", args.size)
    best_bi_lstm = load_best_model("rnn_bi_lstm", args.size)
    best_gru = load_best_model("rnn_gru", args.size)

    to_predict = preprocess_tweets_to_predict()
    _, _, _, _, _, tokenizer, _, _, _ = prepare_data_finetune("full", 200, 100)

    pred_nn = compute_predictions_nn(to_predict, best_nn, tokenizer, maxlen=100)
    pred_cnn = compute_predictions_nn(to_predict, best_cnn, tokenizer, maxlen=100)
    pred_lstm = compute_predictions_nn(to_predict, best_lstm, tokenizer, maxlen=100)
    pred_bi_lstm = compute_predictions_nn(to_predict, best_bi_lstm, tokenizer, maxlen=100)
    pred_gru = compute_predictions_nn(to_predict, best_gru, tokenizer, maxlen=100)

    create_csv_submission(pred_nn, "predictions/pred_nn.csv")
    create_csv_submission(pred_cnn, "predictions/cnn_pred.csv")
    create_csv_submission(pred_lstm, "predictions/pred_lstm.csv")
    create_csv_submission(pred_bi_lstm, "predictions/pred_bi_lstm.csv")
    create_csv_submission(pred_gru, "predictions/pred_gru.csv")

