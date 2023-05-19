import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



#Example Training Set
labelledTweets = {'aino is pog': 1, 'frick aino': 0, 'frick you bino': 0, 'I love honey': 1}


def remove_urls(text):
    """
    Removes URLs from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with URLs removed.
    """
    url_pattern = re.compile(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?')
    return url_pattern.sub('', text)


def preprocess_tweet(tweet):
    """
    Preprocesses a tweet by converting it to lowercase, removing URLs, user mentions,
    numbers, punctuation, tokenizing, removing stop words, and lemmatizing the tokens.

    Args:
        tweet (str): Input tweet.

    Returns:
        list: Preprocessed tokens.
    """
    # Convert to lowercase
    tweet = tweet.lower()
    tweet = remove_urls(tweet)

    # Remove URLs
    tweet = re.sub(r"http\S+", "", tweet)

    # Remove user mentions
    tweet = re.sub(r"@\w+", "", tweet)

    # Remove numbers and punctuation
    tweet = re.sub(r"[^a-zA-Z]", " ", tweet)

    # Tokenize
    tokens = word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token in stop_words]

    # Lemmatize
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def createFreqDict(labelledTweets):
    """
    Creates a frequency dictionary for the given labelled tweets.

    Args:
        labelledTweets (dict): Dictionary of labelled tweets.

    Returns:
        dict: Frequency dictionary where keys are words and values are tuples (pos_count, neg_count).
    """
    freqVocab = {}
    for tweet in labelledTweets:
        words = preprocess_tweet(tweet)
        label = labelledTweets[tweet]
        for word in words:
            if label == 1:
                freqVocab[word] = np.add(freqVocab.get(word, [0, 0]), [1, 0])
            else:
                freqVocab[word] = np.add(freqVocab.get(word, [0, 0]), [0, 1])
    freqVocab['<unk>'] = (1, 1)
    return freqVocab





def GetLogPrior(labelledTweets):
    """
    Calculates the logarithm of the prior probability of positive class (log(P(y=1))).

    Args:
        labelledTweets (dict): Dictionary of labelled tweets.

    Returns:
        float: Logarithm of the prior probability.
    """
    labels = labelledTweets.values()
    Dneg = sum([1 for _ in labels if _ == 0])
    Dpos = sum([1 for _ in labels if _ == 1])
    return np.log(Dpos) - np.log(Dneg)


def getV(freqDict):
    """
    Calculates the size of the vocabulary (number of unique words) in the frequency dictionary.

    Args:
        freqDict (dict): Frequency dictionary.

    Returns:
        int: Size of the vocabulary.
    """
    vocab_size = len(freqDict)
    print(f'V = {vocab_size}')
    return vocab_size


def getN(sentiment, freqDict):
    """
    Calculates the total count of words with the specified sentiment in the frequency dictionary.

    Args:
        sentiment (int): Sentiment label (1 for positive, 0 for negative).
        freqDict (dict): Frequency dictionary.

    Returns:
        int: Total count of words.
    """
    count = sum(value[0] for value in freqDict.values()) if sentiment == 1 else sum(value[1] for value in freqDict.values())
    print(f'N = {count}')
    return count


def getWordProbability(freqDict, word, sentiment):
    """
    Calculates the probability of a word given a sentiment label in the frequency dictionary.

    Args:
        freqDict (dict): Frequency dictionary.
        word (str): Word.
        sentiment (int): Sentiment label (1 for positive, 0 for negative).

    Returns:
        float: Word probability.
    """
    frequency = freqDict[word][0] if sentiment == 1 else freqDict[word][1]
    return (frequency + 1) / (getN(sentiment, freqDict) + getV(freqDict))


def getWordLog(word, freqDict):
    """
    Calculates the logarithm of the ratio of word probabilities for positive and negative classes.

    Args:
        word (str): Word.
        freqDict (dict): Frequency dictionary.

    Returns:
        float: Logarithm of the word ratio.
    """
    return np.log(getWordProbability(freqDict, word, 1) / getWordProbability(freqDict, word, 0))


def PredictDoc(doc, labelledTweets):
    """
    Predicts the sentiment of a document based on the labelled tweets.

    Args:
        doc (str): Document to predict the sentiment for.
        labelledTweets (dict): Dictionary of labelled tweets.

    Returns:
        float: Prediction score.
    """
    freqDict = createFreqDict(labelledTweets)
    tokenized = preprocess_tweet(doc)
    prediction = GetLogPrior(labelledTweets) + sum([getWordLog(token, freqDict) if token in freqDict else getWordLog('<unk>', freqDict) for token in tokenized])
    return prediction


doctest = "This is an example tweet, Honey!"
print(PredictDoc(doctest, labelledTweets))
