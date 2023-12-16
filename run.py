from neural_networks.nn_utils import*
from neural_networks.cnn import*
import argparse


def train_model(model_type, size):
    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len = prepare_data(size)
    if model_type == "simple_nn":
        simple_nn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len)
    
    if model_type == "cnn":
        cnn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, first_dropout=0.4)


def eval_model(model_type, size):
    best_model = load_best_model(model_type)
    _, X_test, _, y_test, _, _, _, _= prepare_data(size)
    evaluate_best_model(model_type, best_model, X_test, y_test)




def main():

    #python run.py --mode train --model_type simple_nn --size small
    #python run.py --mode train --model_type simple_nn --size small
    #python run.py --mode eval --model_type simple_nn --size small
    
    parser = argparse.ArgumentParser(description='Train or Evaluate a Model')
    parser.add_argument('--mode', type=str, required=True, help='Mode to run: "train" or "eval"')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model: e.g., "simple_nn"')
    parser.add_argument('--size', type=str, required=True, help='size of dataset : "small" or "full"')

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.model_type, args.size)
    elif args.mode == "eval":
        eval_model(args.model_type, args.size)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'eval'.")

if __name__ == '__main__':
    main()