import wandb
from subprocess import call
from nn_utils import*
from models_for_finetune import*


def train_simple_nn():
    run = wandb.init()

    # Parameters to be tuned
    dim = run.config.dim
    max_len = run.config.max_len
    batch_size = run.config.batch_size
    lr = run.config.lr

    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, dim = prepare_data_finetune("finetune", dim, max_len)
    
    model, acc = finetune_simple_nn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr)


    run.log({"Test_accuracy": acc})
    run.finish()
    

def train_nn():
    run = wandb.init()

    # Parameters to be tuned
    dim = run.config.dim
    max_len = run.config.max_len
    batch_size = run.config.batch_size
    lr = run.config.lr
    dropout_rate = run.config.dropout_rate
    filters_list = run.config.filters_list
    kernel_sizes = run.config.kernel_sizes
    
    
    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, dim = prepare_data_finetune("finetune", dim, max_len)
    
    #model, acc = finetune_cnn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, first_dropout=0.4)
    model, acc = finetune_cnn(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, dropout_rate, filters_list, kernel_sizes)


    run.log({"Test_accuracy": acc})
    run.finish()


def train_rnn_lstm():
    run = wandb.init()

    # Parameters to be tuned
    dim = run.config.dim
    max_len = run.config.max_len
    batch_size = run.config.batch_size
    lr = run.config.lr
    hidden_units = run.config.hidden_units
    lstm_layers = run.config.lstm_layers
    dropout_rate = run.config.dropout_rate
    recurrent_dropout_rate = run.config.recurrent_dropout_rate

    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, dim = prepare_data_finetune("finetune", dim, max_len)
    
    #model, acc = finetune_rnn_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, first_dropout=0.4)
    model, acc = finetune_rnn_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, hidden_units, lstm_layers, dropout_rate, recurrent_dropout_rate)


    run.log({"Test_accuracy": acc})
    run.finish()


def train_rnn_bi_lstm():
    run = wandb.init()

    # Parameters to be tuned
    dim = run.config.dim
    max_len = run.config.max_len
    batch_size = run.config.batch_size
    lr = run.config.lr
    hidden_units = run.config.hidden_units
    lstm_layers = run.config.lstm_layers
    dropout_rate = run.config.dropout_rate
    recurrent_dropout_rate = run.config.recurrent_dropout_rate


    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, dim = prepare_data_finetune("finetune", dim, max_len)
    
    model, acc = finetune_rnn_bi_lstm(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, hidden_units, lstm_layers, dropout_rate, recurrent_dropout_rate)

    run.log({"Test_accuracy": acc})
    run.finish()


def train_rnn_gru():
    run = wandb.init()

    # Parameters to be tuned
    dim = run.config.dim
    max_len = run.config.max_len
    batch_size = run.config.batch_size
    lr = run.config.lr
    hidden_units = run.config.hidden_units
    lstm_layers = run.config.lstm_layers
    dropout_rate = run.config.dropout_rate



    X_train, X_test, y_train, y_test, vocab_size, tokenizer, embedding_matrix, max_len, dim = prepare_data_finetune("finetune", dim, max_len)
    
    model, acc = finetune_rnn_gru(X_train, y_train, X_test, y_test, vocab_size, embedding_matrix, max_len, dim, batch_size, lr, hidden_units, lstm_layers, dropout_rate)



    run.log({"Test_accuracy": acc})
    run.finish()


# Sweep configuration



sweep_config_cnn = {
    'method': 'grid',
    'metric': {
      'name': 'Test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'dim': {
            'values': [100, 200]  # Example values for CNN
        },

        'lr': {
            'values': [0.0001, 0.0005, 0.001,]  # Example values for CNN
        },

        'max_len': {
            'values': [50, 100]  # Example values for CNN
        },
        'batch_size': {
            'values': [128, 256]  # Example values for CNN
        },
        # Add any other parameters specific to CNN
    }
}

sweep_config_cnn = {
    'method': 'grid',
    'metric': {
      'name': 'Test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'dim': {
            'values': [100, 200]
        },
        'max_len': {
            'values': [50, 100]
        },
        'batch_size': {
            'values': [64, 128]
        },
        'lr': {
            'values': [0.001, 0.0001]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        },
        'filters_list': {
            'values': [[64, 128, 256, 512], [32, 64, 128, 256]]  # Example filter sizes for each layer
        },
        'kernel_sizes': {
            'values': [[3, 3, 3, 3], [5, 5, 5, 5]]  # Example kernel sizes for each layer
        }
    }
}

sweep_config_rnn_lstm = {
    'method': 'grid',
    'metric': {
      'name': 'Test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'dim': {
            'values': [100, 200]
        },
        'max_len': {
            'values': [50, 100]
        },
        'batch_size': {
            'values': [256, 512]
        },
        'lr': {
            'values': [0.0005, 0.0001]
        },
        'hidden_units': {
            'values': [64, 128]
        },
        'lstm_layers': {
            'values': [1, 2]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        },
        'recurrent_dropout_rate': {
            'values': [0.2, 0.5]
        }
    }
}

sweep_config_rnn_gru = {
    'method': 'grid',
    'metric': {
      'name': 'Test_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'dim': {
            'values': [100, 200]
        },
        'max_len': {
            'values': [50, 100]
        },
        'batch_size': {
            'values': [256, 512]
        },
        'lr': {
            'values': [0.0005, 0.0001]
        },
        'hidden_units': {
            'values': [64, 128]
        },
        'lstm_layers': {
            'values': [1, 2]
        },
        'dropout_rate': {
            'values': [0.2, 0.5]
        },
    }
}







def main():
    

    sweep_id = wandb.sweep(sweep_config_cnn, project="ML tweet train_nn")
    wandb.agent(sweep_id, function=lambda: train_nn())

    sweep_id = wandb.sweep(sweep_config_rnn_lstm, project="ML tweet train_rnn_lstm")
    wandb.agent(sweep_id, function=lambda: train_rnn_lstm())

    sweep_id = wandb.sweep(sweep_config_rnn_lstm, project="ML tweet train_rnn_bi_lstm")
    wandb.agent(sweep_id, function=lambda: train_rnn_bi_lstm())

    sweep_id = wandb.sweep(sweep_config_rnn_gru, project="ML tweet train_rnn_gru")
    wandb.agent(sweep_id, function=lambda: train_rnn_gru())


   
if __name__ == '__main__':
    wandb.login(key="ab7685e24cbba84d7b9ee9574c68a4fa7d0ac965")
    main()