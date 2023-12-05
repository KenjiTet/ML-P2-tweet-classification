import re
from .loads import load_tweets

def clean_tweet(tweet):
    """
    Clean a single tweet by removing special characters, URLs, and converting to lowercase.

    Args:
    tweet (str): A single tweet.

    Returns:
    str: Cleaned tweet.
    """
    tweet = tweet.lower()  # Convert to lowercase
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r'\@w+|\#','', tweet)  # Remove @ and #
    tweet = re.sub(r"<user>",'', tweet)  # Remove <user>
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuations
    return tweet

def preprocess_tweets(tweets):
    """
    Apply preprocessing to a list of tweets.

    Args:
    tweets (list): A list of tweets.

    Returns:
    list: A list of cleaned tweets.
    """
    return [clean_tweet(tweet) for tweet in tweets]

def preprocess(input_file_path, output_file_path):
    """
    Load tweets, preprocess them, and write the cleaned tweets to a new file.

    Args:
    input_file_path (str): Path to the input text file containing tweets.
    output_file_path (str): Path to the output text file for cleaned tweets.
    """
    tweets = load_tweets(input_file_path)
    cleaned_tweets = preprocess_tweets(tweets)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for tweet in cleaned_tweets:
            file.write(tweet + '\n')



if __name__ == "__main__":
    preprocess('twitter-datasets/small_pos.txt', 'twitter-datasets/prep_small_pos.txt')
    preprocess('twitter-datasets/small_neg.txt', 'twitter-datasets/prep_small_neg.txt')