from utils.loads import*
from utils.preprocessing import preprocess_tweets
from utils.feature_extraction import construct_features
import subprocess


def generate_word_embeddings():

    """
    Run Shell Scripts:
        build_vocab.sh: Creates a comprehensive vocabulary (vocab_full.txt) from the training dataset, listing all unique words and their frequencies.
        cut_vocab.sh: Refines the vocabulary, retaining only frequently occurring words (more than 4 times) in vocab_cut.

    Run Python Scripts for Word Embeddings:
        pickle_vocab.py: Transforms the refined vocabulary into a Python dictionary, assigning a unique index to each word. The dictionary is then serialized and stored in vocab.pkl for efficient future access.
        cooc.py: Computes the co-occurrence matrix from the tweets and saves it in cooc.pkl. This matrix quantifies the frequency of word pairs occurring together, forming the basis for the GloVe model training.
        glove_solution.py: Conducts the actual GloVe model training. It learns vector representations for each word based on the co-occurrence matrix. The training employs SGD to optimize the embeddings, resulting in vectors (embeddings.npy) that encode the semantic context of words in the dataset.
    """
    
    # Run the shell scripts
    print("-------------Building vocabulary---------------")
    subprocess.run(['python', 'embeddings/build_vocab.py'])

    print("-------------Run Embedding ---------------")
    # Run the Python scripts for word embeddings
    subprocess.run(['python', 'embeddings/pickle_vocab.py'])
    subprocess.run(['python', 'embeddings/cooc.py'])
    subprocess.run(['python', 'embeddings/glove_solution.py'])


def main():

    print("\n---------------- BEGIN MAIN ------------------")
    
    prep_tweet_path = "twitter-datasets/prep_small_pos.txt"
    cleaned_tweets = load_tweets(prep_tweet_path)

    #generate_word_embeddings()  #only need to be done once
    
    
    # Load GloVe embeddings and vocabulary

    embeddings = load_glove_embeddings('resources/embeddings.npy')
    vocab = load_vocabulary('resources/vocab.pkl')

    # Construct feature vectors for training tweets
    features = construct_features(cleaned_tweets, embeddings, vocab)

    
if __name__ == '__main__':
    main()
    
