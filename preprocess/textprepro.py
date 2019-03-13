import re     
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

# def create_vectoriser(vectoriser, stem=False, stop=False):
#     if vectoriser == 'count':
#         return CustomCountVectoriser(stem, stop)
#     elif vectoriser == 'tfidf':
#         return CustomTfidfVectoriser(stem, stop)
#     else:
#         raise

def create_vectoriser(vectoriser):
    if vectoriser == 'count':
        return CountVectorizer(analyzer=str.split)
    elif vectoriser == 'tfidf':
        return TfidfVectorizer(analyzer=str.split)
    else:
        raise


def myPreprocesser(doc):
    doc = doc.lower()
    doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)
    return doc

class CustomCountVectoriser(CountVectorizer):
    def __init__(self, stem=False, stop=True):
        CountVectorizer.__init__(self)
        self.stem = stem
        self.stop = stop
    def build_preprocessor(self):
        preprocessor = super(CustomCountVectoriser, self).build_preprocessor()
        return lambda doc: myPreprocesser(doc)
    def build_tokenizer(self):
        tokeniser = super(CustomCountVectoriser, self).build_tokenizer()
        if self.stem:
            if self.stop:
                return lambda doc: [PorterStemmer().stem(t) for t in tokeniser(doc) if t not in stopwords.words('english')]
            else:
                return lambda doc: [PorterStemmer().stem(t) for t in tokeniser(doc)]
        else:
            if self.stop:
                return lambda doc: [t for t in tokeniser(doc) if t not in stopwords.words('english')]
            else:
                return lambda doc: list(tokeniser(doc))

class CustomTfidfVectoriser(TfidfVectorizer):
    def __init__(self, stem=False, stop=True):
        TfidfVectorizer.__init__(self)
        self.stem = stem
        self.stop = stop
    def build_preprocessor(self):
        preprocessor = super(CustomTfidfVectoriser, self).build_preprocessor()
        return lambda doc: myPreprocesser(doc)
    def build_tokenizer(self):
        tokeniser = super(CustomTfidfVectoriser, self).build_tokenizer()
        if self.stem:
            if self.stop:
                return lambda doc: [PorterStemmer().stem(t) for t in tqdm(tokeniser(doc)) if t not in stopwords.words('english')]
            else:
                return lambda doc: [PorterStemmer().stem(t) for t in tqdm(tokeniser(doc))]
        else:
            if self.stop:
                return lambda doc: [t for t in tqdm(tokeniser(doc)) if t not in stopwords.words('english')]
            else:
                return lambda doc: list(tokeniser(doc))