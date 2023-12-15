import re
from .loads import*
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from sklearn import metrics
import nltk
nltk.download('wordnet')


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

def preprocess(input_file_path, output_file_path, tweets):
    """
    Load tweets, preprocess them, and write the cleaned tweets to a new file.

    Args:
    input_file_path (str): Path to the input text file containing tweets.
    output_file_path (str): Path to the output text file for cleaned tweets.
    """
    cleaned_tweets = preprocess_tweets(tweets)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for tweet in cleaned_tweets:
            file.write(tweet + '\n')


def combine_dataset(path1, path2):
    """
    Combine tweets from two text files into a single list.

    Args:
    path1 (str): Path to the first text file.
    path2 (str): Path to the second text file.

    Returns:
    list: A list containing tweets from both files.
    """

    tweets = []

    # Read and add tweets from the first file
    with open(path1, 'r', encoding='utf-8') as file:
        tweets.extend(file.readlines())

    # Read and add tweets from the second file
    with open(path2, 'r', encoding='utf-8') as file:
        tweets.extend(file.readlines())

    return tweets
    

def replace_exclamations(x):
    """ Replaces multiple exclamation marks by the word exclamationMark """
    x = re.sub('(\! )+(?=(\!))', '', x)
    x = re.sub(r"(\!)+", ' exclamationMark ', x)
    return x

def replace_points(x):
    """ Replaces multiple points by the word multiplePoints """
    x = re.sub('(\. )+(?=(\.))', '', x)
    x = re.sub(r"(\.)+", ' multistop ', x)
    return x


def replace_questions(x):
    """ Replaces multiple question marks by the word questionMark """
    x = re.sub('(\? )+(?=(\?))', '', x)
    x = re.sub(r"(\?)+", ' questionMark ', x)
    return x

def tokenization(text):
    text = re.split('\W+', text)
    return text

def lemmatizer(l,text):
    text = [l.lemmatize(word) for word in text]
    return text

def join_tokens(tokens):
    text = ' '.join(tokens)
    return text

def translate_emojis(x):
    """ Replace emojis into meaningful words """
    
    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\),\],},D,>,3,d] ', ' happy ', x) #:) :D :} :] :> :') :'D :-D :-)
    x = re.sub(' [\(,\[,{,<]( )*([\',\"])*( )*[:,=,8,;] ', ' happy ', x) #inverted happy 
    x = re.sub(' [X,x]( )*D ', ' funny ', x) #XD xD
    x = re.sub(' ^( )*[.,~]( )*^ ', ' happy ', x) #^.^  ^ ~ ^  
    x = re.sub(' ^( )*(_)+( )*^ ', ' happy ', x) #^__^
    
    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\(,\[,{,<] ', ' sad ', x) #:( :{ :[ :< :'(
    x = re.sub(' [\),\],},D,>,d]( )*(-)*( )*([\',\"])*( )*[:,=,8,;] ', ' sad ', x) #inverted sad
    x = re.sub(' >( )*.( )*< ', ' sad ', x) #>.<
    x = re.sub(' <( )*[ \/,\\ ]( )*3 ', ' sad ', x) #</3
    
    x = re.sub(' [:,=,8]( )*(-)*( )*p ', ' silly ', x) #:p :-p
    x = re.sub(' q( )*(-)*( )*[:,=,8] ', ' silly ', x) #silly inverted
    
    x = re.sub(' [:,=,8]( )*$ ', ' confused ', x) #=$ 8$
    x = re.sub(' [:,=,8]( )*@ ', ' mad ', x) #:@
    x = re.sub(' [:,=,8]( )*(-)*( )*[\/,\\,|] ', ' confused ', x) #:/ :\
    x = re.sub(' [\/,\\,|]( )*(-)*( )*[:,=,8] ', ' confused ', x) #confused inverted
    
    x = re.sub(' [:,=,8,;]( )*(-)*( )*[o,0] ', ' surprised ', x) #:o :-O
    
    x = re.sub(' [x,X]+ ', ' kiss ', x) #xXxX
    x = re.sub(' ([x,X][o,O]){2,} ', ' kiss ', x) #xoxo
    x = re.sub(' [:,=,8,;]( )*\* ', ' kiss ', x) #:* =*
    x = re.sub(' <( )*3 ', ' love ', x) #<3
    
    x = re.sub('#', ' hashtag ', x) #hashtag
    x = re.sub('&', ' and ', x) #&
    x = re.sub(' \(( )*y( )*\) ', ' yes ', x) #(y)
    x = re.sub(' w( )*/ ', ' without ', x) #w/
    
    x = re.sub(' ([h,j][a,e,i,o]){2,} ', ' haha ', x) #hahah
    x = re.sub(' (a*ha+h[ha]*|h*ah+a[ah]*|o?l+o+l+[ol]*) ', ' haha ', x) #hhaha,aahha,lool
    x = re.sub(' (i*hi+h[hi]*|h*ih+i[ih]*|h*oh+o[oh]*|h*eh+e[eh]*) ', ' haha ', x) #hihi, hoho, hehe
    return x

def split_negation(text):
    negations_dict = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                    "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                    "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                    "mustn't":"must not","ain't":"is not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dict.keys()) + r')\b')
    text = neg_pattern.sub(lambda x: negations_dict[x.group()], text)
    return text

def replace_contractions(text):
    contractions_dict = {"i'm":"i am", "wanna":"want to", "whi":"why", "gonna":"going to",
                    "wa":"was","nite":"night","there's":"there is","that's":"that is",
                    "ladi":"lady", "fav":"favorite", "becaus":"because","i\'ts":"it is",
                    "dammit":"damn it", "coz":"because", "ya":"you", "dunno": "do not know",
                    "donno":"do not know","donnow":"do not know","gimme":"give me"}
    contraction_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    text = contraction_pattern.sub(lambda x: contractions_dict[x.group()], text)
    
    contraction_patterns = [(r'ew(\w+)', 'disgusting'),(r'argh(\w+)', 'argh'),(r'fack(\w+)', 'fuck'),
                            (r'sigh(\w+)', 'sigh'),(r'fuck(\w+)', 'fuck'),(r'omg(\w+)', 'omg'),
                            (r'oh my god(\w+)', 'omg'),(r'(\w+)n\'', '\g<1>ng'),(r'(\w+)n \'', '\g<1>ng'),
                            (r'(\w+)\'ll', '\g<1> will'),(r'(\w+)\'ve', '\g<1> have'),(r'(\w+)\'s', '\g<1> is'),
                            (r'(\w+)\'re', '\g<1> are'),(r'(\w+)\'d', '\g<1> would'),(r'&', 'and'),
                            ('y+a+y+', 'yay'),('y+[e,a]+s+', 'yes'),('n+o+', 'no'),('a+h+','ah'),('m+u+a+h+','kiss'),
                            (' y+u+p+ ', ' yes '),(' y+e+p+ ', ' yes '),(' idk ',' i do not know '),(' ima ', ' i am going to '),
                            (' nd ',' and '),(' dem ',' them '),(' n+a+h+ ', ' no '),(' n+a+ ', ' no '),(' w+o+w+', 'wow '),
                            (' w+o+a+ ', ' wow '),(' w+o+ ', ' wow '),(' a+w+ ', ' cute '), (' lmao ', ' haha '),(' gad ', ' god ')]
    patterns = [(re.compile(regex_exp, re.IGNORECASE), replacement)
                for (regex_exp, replacement) in contraction_patterns]
    for (pattern, replacement) in patterns:
        (text, _) = re.subn(pattern, replacement, text)
    return text

def tweet_cleaner(tweet):
    #Add trailing and leading whitespaces for the sake of preprocessing
    tweet = ' '+tweet+' '
    #translate emojis
    tweet = translate_emojis(tweet)
    #lowercase tweet
    tweet = tweet.lower()
    #seperate negation words
    tweet = split_negation(tweet)
    #seperate punctuation from words
    tweet = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", tweet)
    #remove the observed pattern of numbers seen above
    tweet = re.sub(r'\\ [0-9]+ ', '', tweet)
    #replace ?,!,. by words
    tweet = replace_exclamations(tweet)
    tweet = replace_questions(tweet)
    tweet = replace_points(tweet)
    
    #Now since we translated punctuation and emojis and negative words we can remove the rest of the 'unwanted' chars
    #remove unwanted punctuation
    tweet = re.sub("[^a-zA-Z]", " ", tweet)
    
    #remove trailing and leading whitespace
    tweet = tweet.strip() 
    #remove multiple consecutive whitespaces
    tweet = re.sub(' +', ' ',tweet) 
    
    #Lemmatization
    l = WordNetLemmatizer() 
    tweet = tokenization(tweet)
    tweet = join_tokens(lemmatizer(l,tweet))
    return tweet

def preprocess_tweets():

    train_set, _ = train_test_cleaner()

    with open("resources/tweet.pkl", "wb") as f:
        pickle.dump(train_set, f)  


def train_test_cleaner():
	"""Clean train set and test set and return cleaned dataframes"""
	#Read positive tweets train file
	with open('twitter-datasets/small_pos.txt',"r",encoding="utf8") as file:
		train_pos = file.read().split('\n')
	train_pos = pd.DataFrame({'tweet' : train_pos})[:len(train_pos)-1]
	#Read negative tweets train file
	with open('twitter-datasets/small_neg.txt',"r",encoding="utf8") as file:
		train_neg = file.read().split('\n')
	train_neg = pd.DataFrame({'tweet' : train_neg})[:len(train_neg)-1]
	#Read test tweets file
	with open('twitter-datasets/test_data.txt',"r",encoding="utf8") as file:
		df_unknown = file.read().split('\n')
	df_unknown = pd.DataFrame({'tweet' : df_unknown})[:len(df_unknown)-1]
	df_unknown.index += 1 
	df_unknown['tweet'] = df_unknown['tweet'].apply(lambda x : str(x).split(',', maxsplit=1)[1])
		
	#Drop duplicates
	train_neg.drop_duplicates(inplace=True)
	train_pos.drop_duplicates(inplace=True)
	
	#Add labels
	train_pos['label'] = 1
	train_neg['label'] = 0
	train_set = pd.concat([train_pos, train_neg], ignore_index=True)

	#Apply tweet cleaner to train set and test
	train_set['tweet'] = train_set['tweet'].apply(tweet_cleaner)
	df_unknown['tweet'] = df_unknown['tweet'].apply(tweet_cleaner)
	
	return train_set,df_unknown
	

def train_test_df():
	"""Clean train set and test set and return cleaned dataframes"""
     
	#Read positive tweets train file
	with open('twitter-datasets/prep_train_pos_full.txt',"r",encoding="utf8") as file:
		train_pos = file.read().split('\n')
	train_pos = pd.DataFrame({'tweet' : train_pos})[:len(train_pos)-1]
     

	#Read negative tweets train file
	with open('twitter-datasets/prep_train_neg_full.txt',"r",encoding="utf8") as file:
		train_neg = file.read().split('\n')
	train_neg = pd.DataFrame({'tweet' : train_neg})[:len(train_neg)-1]
     

	#Read test tweets file
	with open('twitter-datasets/test_data.txt',"r",encoding="utf8") as file:
		df_unknown = file.read().split('\n')
	df_unknown = pd.DataFrame({'tweet' : df_unknown})[:len(df_unknown)-1]
	df_unknown.index += 1 
	df_unknown['tweet'] = df_unknown['tweet'].apply(lambda x : str(x).split(',', maxsplit=1)[1])
		

	#Drop duplicates
	train_neg.drop_duplicates(inplace=True)
	train_pos.drop_duplicates(inplace=True)
	
	#Add labels
	train_pos['label'] = 1
	train_neg['label'] = -1
     
	train_set = pd.concat([train_pos, train_neg], ignore_index=True)

	
	return train_set, df_unknown




    


