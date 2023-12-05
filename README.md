# ML-P2-tweet-classification
ML-P2-Tweet-classification/
│
├── twitter-datasets/
│   ├── train_neg_full.txt
│   ├── train_neg.txt
│   ├── train_pos_full.txt
│   ├── train_pos.txt
│   └── test_data.txt
│
├── data_resources/ 
│   ├── vocab_full.txt
│   ├── vocab_cut.txt
│   ├── vocab.pkl
│   ├── embeddings.npy
│   └── cooc.pkl
│
├── embeddings/ 
│   ├── pickle_vocab.py
│   ├── cooc.py
│   ├── glove_template.py
│   ├── glove_solution.py
│   └── build_vocab.py
│
├── utils/
│   ├── preprocessing.py 
│   └── feature_extraction.py 
│
├── notebooks/ 
│   └── EDA.ipynb
│
├── run.py 
├── requirements.txt 
└── README.md 



# Tweet Classifier Project

## Overview
This project involves a machine learning application to classify tweets as positive or negative. It uses Python for preprocessing the data, extracting features, and building a classifier model.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)


## Running the Project
To run the tweet classification project, follow these steps:

1. **Install requirements**
run the following: 
pip install -r requirements.txt

2. **Preprocess the Data:**
The project requires preprocessed tweet data. If you have not preprocessed your data, follow the instructions in `utils/preprocessing.py`.

3. **Feature Extraction:**
Ensure your data is ready for feature extraction as outlined in `utils/feature_extraction.py`.

4. **Execute the Main Script:**
Run the `run.py` script from the project's root directory: 
python run.py

