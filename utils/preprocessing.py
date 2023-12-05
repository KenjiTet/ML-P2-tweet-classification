import re

def load_tweets(file_path):
    """
    Load tweets from a text file.

    Args:
    file_path (str): Path to the text file containing tweets.

    Returns:
    list: A list of tweets.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

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
