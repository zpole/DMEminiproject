import re
from nltk import word_tokenize          
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

def create_vectoriser(vectoriser, stem=False, stop=False):
    if stem:
        tokeniser = PorterTokenizer()
    else:
        tokeniser = None

    if stop:
        stop_words = 'english'
    else:
        stop_words = None

    if vectoriser == 'count':
        return CountVectorizer(
            preprocessor = myPreprocesser,
            tokenizer = tokeniser,
            token_pattern = '\\b\\w+\\b',
            stop_words = stop_words
        )
    elif vectoriser == 'tfidf':
        return TfidfVectorizer(
            preprocessor = myPreprocesser,
            tokenizer = tokeniser,
            token_pattern = '\\b\\w+\\b',
            stop_words = stop_words
        )

def myPreprocesser(doc):
    doc = doc.lower()
    doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)
    return doc

class PorterTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) if t not in list(string.punctuation)]
