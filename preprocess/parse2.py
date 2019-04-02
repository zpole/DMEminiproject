from html.parser import HTMLParser
import glob
import re
import os
import argparse
from tqdm import tqdm
import unidecode
# import preprocess.textprepro
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
from nltk.corpus import stopwords

def getRootDir(stem, stop):
    if stem:
        if stop:
            root_dir = 'tokenstemstop'
        else:
            root_dir = 'tokenstem'
    else:
        if stop:
            root_dir = 'tokenstop'
        else:
            root_dir = 'tokens'
    
    return root_dir

class MyHTMLParser(HTMLParser):
    def __init__(self, txt, stem=False, stop=False):
        HTMLParser.__init__(self)
        self.txt = txt
        self.stem = stem
        self.stop = stop
        self.tokeniser = CountVectorizer(token_pattern=r'(?u)\b\w+\b').build_tokenizer()
        self.root = False

    def handle_starttag(self, tag, attrs):
        if self.root:
            self.root = False
        # print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        pass
        # print("Encountered an end tag :", tag)

    def handle_data(self, doc):
        if not self.root:
            doc = unidecode.unidecode(doc)
            doc = doc.lower()
            doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)

            doc = re.sub(r'(?!\D)\d{1,2}:\d\d:\d\d(?=\D)', ' PlHolderTime ', doc)
            doc = re.sub(r'(?!\D)\d{1,2}:\d\d(?=\D)', ' PlHolderTime ', doc)

            doc = re.sub(r'(?!\D)\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
            doc = re.sub(r'(?!\D)\d{3}\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
            doc = re.sub(r'(?!\D\D)\d{3}\D\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)

            doc = re.sub(r'(?!\D)\d{2,}x\d{2,}(?=\D)', ' PlHolderResolution ', doc)

            if self.stem:
                if self.stop:
                    doc = [PorterStemmer().stem(t) for t in self.tokeniser(doc) if t not in stopwords.words('english')]
                else:
                    doc = [PorterStemmer().stem(t) for t in self.tokeniser(doc)]
            else:
                if self.stop:
                    doc = [t for t in self.tokeniser(doc) if t not in stopwords.words('english')]
                else:
                    doc = list(self.tokeniser(doc))
            if len(doc):
                print(' '.join(doc), file=self.txt, end=' ')

def prepro(doc, tokeniser, stem=False, stop=False):
    # doc = unidecode.unidecode(doc)
    # doc = doc.lower()
    doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)

    doc = re.sub(r'(?!\D)\d{1,2}:\d\d:\d\d(?=\D)', ' PlHolderTime ', doc)
    doc = re.sub(r'(?!\D)\d{1,2}:\d\d(?=\D)', ' PlHolderTime ', doc)

    doc = re.sub(r'(?!\D)\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
    doc = re.sub(r'(?!\D)\d{3}\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
    doc = re.sub(r'(?!\D\D)\d{3}\D\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)

    doc = re.sub(r'(?!\D)\d{2,}x\d{2,}(?=\D)', ' PlHolderResolution ', doc)

    if stem:
        if stop:
            doc = [PorterStemmer().stem(t) for t in tokeniser(doc) if t not in stopwords.words('english')]
        else:
            doc = [PorterStemmer().stem(t) for t in tokeniser(doc)]
    else:
        if stop:
            doc = [t for t in tokeniser(doc) if t not in stopwords.words('english')]
        else:
            doc = list(tokeniser(doc))
    return doc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='some text here')
    parser.add_argument('--stem', type=str, default='False', help='Stemming')
    parser.add_argument('--stop', type=str, default='False', help='Stopping')

    args = parser.parse_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
        else:
            raise
    root_dir = getRootDir(stem=args.stem, stop=args.stop)
    os.makedirs(root_dir)

    this_file = None
    
    tokeniser = CountVectorizer(token_pattern=r'(?u)\b\w+\b').build_tokenizer()
    
    for line in tqdm(open('rmtag.txt')):
        ln = line.strip()
        if ln[:2]=='<<' and ln[-2:]=='>>':
            filename = ln[2:-2]
            label, uni, name = filename.split(os.sep)[1:]
            os.makedirs(os.path.join(root_dir, label, uni), exist_ok=True)
        else:
            with open(os.path.join(root_dir, label, uni, name+'.txt'), 'a+', encoding='utf-8') as txt:
                print(' '.join(prepro(ln, tokeniser, stem=args.stem, stop=args.stop)), file=txt, end=' ')
        

    # allfiles = glob.glob('webkb/**/*', recursive=True)
    
    # for file in tqdm(allfiles):
    #     try:
    #         html = open(file).read()
    #     except UnicodeDecodeError:
    #         html = open(file, encoding='iso-8859-1').read()
    #     except IsADirectoryError:
    #         continue
    #     except:
    #         print(file)
    #         raise

    #     label, uni, name = file.strip().split('/')[1:]
    #     os.makedirs(os.path.join(root_dir, label, uni), exist_ok=True)
    #     with open(os.path.join(root_dir, label, uni, name+'.txt'), 'a+', encoding='utf-8') as txt:
    #         parser = MyHTMLParser(txt, stem=args.stem, stop=args.stop)
    #         parser.feed(html)