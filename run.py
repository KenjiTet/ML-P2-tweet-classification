from utils.preprocessing import load_tweets, preprocess_tweets
import subprocess




def preprocess():
    file_path = 'twitter-datasets/small_pos.txt'  # Path to your dataset
    tweets = load_tweets(file_path)
    cleaned_tweets = preprocess_tweets(tweets)

    print(f"Exemple of clean tweet: {cleaned_tweets[0]}\n")


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
    preprocess()
    generate_word_embeddings()

    
if __name__ == '__main__':
    main()
    
