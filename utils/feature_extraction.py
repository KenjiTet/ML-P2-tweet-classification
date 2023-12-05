import numpy as np



def tweet_to_feature(tweet, embeddings, vocab):
    """
    Convert a tweet into a feature vector by averaging its word embeddings.
    """
    words = tweet.split()
    word_vectors = [embeddings[vocab[word]] for word in words if word in vocab]
    if not word_vectors:  # Handle case with no words found in vocab
        return np.zeros(embeddings.shape[1])
    return np.mean(word_vectors, axis=0)


def construct_features(tweets, embeddings, vocab):
    """
    Construct feature vectors for a list of tweets.
    """
    return np.array([tweet_to_feature(tweet, embeddings, vocab) for tweet in tweets])
