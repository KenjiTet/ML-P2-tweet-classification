# ML-P2-tweet-classification
```
ML-P2-Tweet-classification/
|
├── embeddings/
│   └── embedding_training.py
|
├── models/
│   ├── baseline.py
│   ├── bayes.py
│   ├── log_reg.py
│   ├── sgd.py
│   ├── SVC.py
│   └── model_utils.py
|
├── neural_networks/
│   ├── best_models_saved/
|   |   └── best_{model_type}_{size}.h5
│   ├── cnn.py
│   ├── finetune.py
│   ├── models_for_finetune.py
│   ├── nn_utils.py
│   └── rnn.py
|
├── notebooks/
│   └── EDA.ipynb
|
├── predictions/
│   ├── final_pred/
|   |   └── majority_vote_preds.csv
│   └── pred_{model_type}.csv
|
├── resources/
│   ├── trained_w2v_embeddings_{size}_{dim}.txt
│   └── tweet_{size}.pkl
|
├── twitter-datasets/
│   ├── train_neg_full.txt 
│   ├── train_neg.txt 
│   ├── train_pos_full.txt
│   ├── train_pos.txt
│   ├── small_neg.txt
│   ├── small_pos.txt
│   └── test_data.txt
|
├── preprocessing.py
├── requirements.txt
├── README.md
├── run.py
├── setup.py
└── train_eval.py
```



# Tweet Classifier Project

## Overview
This project involves a machine learning application to classify tweets as positive or negative. It uses Python for preprocessing the data, extracting features, and building a classifier model.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

**Install requirements**
run the following: 
pip install -r requirements.txt

## Train embedding and create the tweets.pkl
To be able to train and evaluate the different models, you will first need to run the setup.py file.
This file will create the desired resources in the resources folder, that are the learned embedding and the tweet.pkl

run the following command : python setup.py --size <size> --dim <dim>

The possible sizes are : full, medium or small
The possible dims are : 200 or 100

exemple : python setup.py --size small --dim 200

The resulting files should be created :
```
├── resources/
│   ├── trained_w2v_embeddings_{size}_{dim}.txt
│   └── tweet_{size}.pkl
```


## Train and evaluate the models
Once the resources successfully created you can use the train_eval.py to train or evaluate the desired neural network model. Running this file will automatically save the trained model if it achieved a better accuracy than the previous saved model of the same type.

To train:
run the following command : python train_eval.py --mode train --model_type <model_type> --size <size>

To evaluate:
run the following command : python train_eval.py --mode eval --model_type <model_type> --size <size>

The possible sizes are : full, medium or small
The possible model_type are : simple_nn, cnn, lstm, bi-lstm, gru

example: python train_eval.py --mode train --model_type simple_nn --size small

**Important**
Before selecting the size you must have ran the setup.py file using the same size

## Training and evaluating the BERT model
The BERT model was specially designed to run on the notebook BertSubmission.ipynb.

Because of the high computational requirement of this model, it was designed to run on google collab to take advantage of the added computational power from the GPU. So in order to run it, please load the notebook on your own collab environment.

To train:
The BertSubmission.ipynb contains two code cells, the first one is used to train the BERT model, after which it saves the model in a folder.

To evaluate:
run the second code cell in BertSubmission.ipynb, this cell loads the model and generates a submission.csv file to evaluate the model's accuracy.

**Note:**
You may need to adjust the variable `folder_path` to match your own environment.

## Creating the final submission file
Once all the models have been trained using the size full, simply run the run.py file to create the final
majority_vote_preds.csv
```
├── predictions/
│   ├── final_pred/
|   |   └── majority_vote_preds.csv
```
