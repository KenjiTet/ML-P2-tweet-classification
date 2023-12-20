import re
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer 
import nltk
nltk.download('wordnet')



def translate_emojis(x):
    """ Replace emojis into words """
    
    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\),\],},D,>,3,d] ', ' happy ', x) 
    x = re.sub(' [\(,\[,{,<]( )*([\',\"])*( )*[:,=,8,;] ', ' happy ', x)
    x = re.sub(' [X,x]( )*D ', ' funny ', x) 
    x = re.sub(' ^( )*[.,~]( )*^ ', ' happy ', x)
    x = re.sub(' ^( )*(_)+( )*^ ', ' happy ', x)
    
    x = re.sub(' [:,=,8,;]( )*([\',\"])*( )*(-)*( )*[\(,\[,{,<] ', ' sad ', x) 
    x = re.sub(' [\),\],},D,>,d]( )*(-)*( )*([\',\"])*( )*[:,=,8,;] ', ' sad ', x) 
    x = re.sub(' >( )*.( )*< ', ' sad ', x)
    x = re.sub(' <( )*[ \/,\\ ]( )*3 ', ' sad ', x) 
    
    x = re.sub(' [:,=,8]( )*(-)*( )*p ', ' silly ', x) 
    x = re.sub(' q( )*(-)*( )*[:,=,8] ', ' silly ', x) 
    
    x = re.sub(' [:,=,8]( )*$ ', ' confused ', x) 
    x = re.sub(' [:,=,8]( )*@ ', ' mad ', x) 
    x = re.sub(' [:,=,8]( )*(-)*( )*[\/,\\,|] ', ' confused ', x) 
    x = re.sub(' [\/,\\,|]( )*(-)*( )*[:,=,8] ', ' confused ', x) 
    
    x = re.sub(' [:,=,8,;]( )*(-)*( )*[o,0] ', ' surprised ', x) 
    
    x = re.sub(' [x,X]+ ', ' kiss ', x) 
    x = re.sub(' ([x,X][o,O]){2,} ', ' kiss ', x) 
    x = re.sub(' [:,=,8,;]( )*\* ', ' kiss ', x) 
    x = re.sub(' <( )*3 ', ' love ', x) 
    
    x = re.sub('#', ' hashtag ', x) 
    x = re.sub('&', ' and ', x) 
    x = re.sub(' \(( )*y( )*\) ', ' yes ', x)
    x = re.sub(' w( )*/ ', ' without ', x) 
    
    x = re.sub(' ([h,j][a,e,i,o]){2,} ', ' haha ', x) 
    x = re.sub(' (a*ha+h[ha]*|h*ah+a[ah]*|o?l+o+l+[ol]*) ', ' haha ', x) 
    x = re.sub(' (i*hi+h[hi]*|h*ih+i[ih]*|h*oh+o[oh]*|h*eh+e[eh]*) ', ' haha ', x) 
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



def replace_ponctuation(x):
    """ Replaces ponctuation by words """
    x = re.sub('(\! )+(?=(\!))', '', x)
    x = re.sub(r"(\!)+", ' exclamationMark ', x)
    x = re.sub('(\. )+(?=(\.))', '', x)
    x = re.sub(r"(\.)+", ' multistop ', x)
    x = re.sub('(\? )+(?=(\?))', '', x)
    x = re.sub(r"(\?)+", ' questionMark ', x)
    return x

def lemmatizer(text, l):
    # Tokenize
    tokens = re.split('\W+', text)
    # Lemmatize
    lemmatized_tokens = [l.lemmatize(token) for token in tokens]
    # Join tokens
    return ' '.join(lemmatized_tokens)


def remove_unwanted_char(tweet):
    tweet = re.sub("[^a-zA-Z]", " ", tweet)
    tweet = tweet.strip() 
    tweet = re.sub(' +', ' ',tweet) 

    return tweet

def separate_punctuation(tweet):
    tweet = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", tweet)
    tweet = re.sub(r'\\ [0-9]+ ', '', tweet)
    return tweet

def tweet_cleaner(tweet):
    tweet = ' '+tweet+' '
    tweet = translate_emojis(tweet)
    tweet = tweet.lower()
    tweet = split_negation(tweet)
    tweet = separate_punctuation(tweet)
    tweet = replace_ponctuation(tweet)
    tweet = remove_unwanted_char(tweet)
    
    l = WordNetLemmatizer() 
    tweet = lemmatizer(tweet, l)
    
    return tweet


def clean_train_test(size):

    if size == "full":
         neg_path = 'twitter-datasets/train_neg_full.txt'
         pos_path = 'twitter-datasets/train_pos_full.txt'

    elif size == "medium":
         neg_path = 'twitter-datasets/train_neg.txt'
         pos_path = 'twitter-datasets/train_pos.txt'  

    elif size == "finetune":
         neg_path = 'twitter-datasets/train_neg_finetune.txt'
         pos_path = 'twitter-datasets/train_pos_finetune.txt'  

    else:
         neg_path = 'twitter-datasets/small_neg.txt'
         pos_path = 'twitter-datasets/small_pos.txt'

    

    """Clean train set and test set and return cleaned dataframes"""

    # Read positive tweets train file
    with open(pos_path, "r", encoding="utf8") as file:
        train_pos = file.read().split('\n')
    train_pos = pd.DataFrame({'tweet': train_pos})[:len(train_pos)-1]

    # Read negative tweets train file
    with open(neg_path, "r", encoding="utf8") as file:
        train_neg = file.read().split('\n')
    train_neg = pd.DataFrame({'tweet': train_neg})[:len(train_neg)-1]

    # Read test tweets file
    with open('twitter-datasets/test_data.txt', "r", encoding="utf8") as file:
        df_unknown = file.read().split('\n')
    df_unknown = pd.DataFrame({'tweet': df_unknown})[:len(df_unknown)-1]
    df_unknown.index += 1 
    df_unknown['tweet'] = df_unknown['tweet'].apply(lambda x: str(x).split(',', maxsplit=1)[1])

    # Drop duplicates
    train_neg.drop_duplicates(inplace=True)
    train_pos.drop_duplicates(inplace=True)

    # Add labels
    train_pos['label'] = 1
    train_neg['label'] = 0
    train_set = pd.concat([train_pos, train_neg], ignore_index=True)

    # Apply tweet cleaner to train set and test set
    train_set['tweet'] = train_set['tweet'].apply(tweet_cleaner)
    df_unknown['tweet'] = df_unknown['tweet'].apply(tweet_cleaner)

    return train_set, df_unknown


def preprocess_tweets(size):

    train_set, _ = clean_train_test(size)

    with open(f"resources/tweet_{size}_test.pkl", "wb") as f:
        pickle.dump(train_set, f)  


def preprocess_tweets_to_predict():
    _, to_predict = clean_train_test(size="small")

    return to_predict






    


