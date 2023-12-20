from neural_networks.nn_utils import*
from neural_networks.cnn import*
from neural_networks.rnn import*
import argparse



def train_model(model_type, size):

    if model_type == "simple_nn":
        max_len = 100
        embedding_dim = 200
        X_train, X_test, y_train, y_test, vocab_size, _, embedding_matrix, _, _ = prepare_data(size, embedding_dim, max_len)
        simple_nn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim=embedding_dim, lr=0.001)
    
    elif model_type == "cnn":
        max_len = 100
        embedding_dim = 200
        X_train, X_test, y_train, y_test, vocab_size, _, embedding_matrix, _, _ = prepare_data(size, embedding_dim, max_len)
        cnn(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim=embedding_dim, dropout_rate= 0.5, filters_list=[64,128,256,512], kernel_sizes= [5,5,5,5], lr = 0.001)

    elif model_type == "lstm":
        max_len = 50
        embedding_dim = 200
        X_train, X_test, y_train, y_test, vocab_size, _, embedding_matrix, _, _ = prepare_data(size, embedding_dim, max_len)
        rnn_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim=embedding_dim, lr= 0.0005, hidden_units= 128, lstm_layers= 2, dropout_rate= 0.2, recurrent_dropout_rate= 0.5)

    elif model_type == "bi-lstm":
        max_len = 100
        embedding_dim = 200
        X_train, X_test, y_train, y_test, vocab_size, _, embedding_matrix, _, _ = prepare_data(size, embedding_dim, max_len)
        rnn_bi_lstm(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim=embedding_dim, lr=0.0005, hidden_units=128, bi_lstm_layers=2, dropout_rate=0.2, recurrent_dropout_rate=0.5)

    elif model_type == "gru":
        max_len = 100
        embedding_dim = 200
        X_train, X_test, y_train, y_test, vocab_size, _, embedding_matrix, _, _ = prepare_data(size, embedding_dim, max_len)
        rnn_gru(size, X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim=embedding_dim, lr=0.0005, hidden_units=128, gru_layers=1, dropout_rate=0.3, batch_size=BATCH_SIZE)

    else:
        print(f"The model_type is incorrect or invalid, select one of the following:\n-simple_nn\n-cnn\n-lstm\n-bi-lstm\n-gru")



def eval_model(model_type, size):
    dim = 100

    if model_type == "lstm":
        dim = 50

    best_model = load_best_model(model_type, size)
    _, X_test, _, y_test, _, _, _, _, _ = prepare_data(size, 200, dim)

    evaluate_best_model(model_type, best_model, X_test, y_test)




def main():

    #python train_eval.py --mode train --model_type simple_nn --size small
    #python train_eval.py --mode train --model_type simple_nn --size small
    #python train_eval.py --mode eval --model_type simple_nn --size small
    
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