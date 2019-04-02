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

def getRootSuffix(args):
    suffix = '_'
    suffix += '1' if args.stop else '0'
    suffix += '1' if args.stem else '0'
    suffix += '1' if args.mime else '0'
    suffix += '1' if args.digit else '0'
    suffix += '1' if args.other else '0'
    return suffix

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
    def __init__(self, txt, args):
        HTMLParser.__init__(self)
        self.txt = txt
        self.args = args
        self.tokeniser = CountVectorizer(token_pattern=r'(?u)\b\w+\b').build_tokenizer()

        if self.args.mime:
            self.root = True
        else:
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

            if self.args.other:
                doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)
            
            if self.args.digit:
                doc = re.sub(r'(?!\D)\d{1,2}:\d\d:\d\d(?=\D)', ' PlHolderTime ', doc)
                doc = re.sub(r'(?!\D)\d{1,2}:\d\d(?=\D)', ' PlHolderTime ', doc)

                doc = re.sub(r'(?!\D)\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
                doc = re.sub(r'(?!\D)\d{3}\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
                doc = re.sub(r'(?!\D\D)\d{3}\D\D\d{3}\-\d{4}(?=\D)', ' PlHolderPhoneNum ', doc)
                
                doc = re.sub(r'(?!\D)\d{5}-\d{4}(?=\D)', ' PlHolderZipPlusFour ', doc)
                
                doc = re.sub(r'(?!\D)\d{2,}x\d{2,}x\d{2,}(?=\D)', ' PlHolderResolution ', doc)
                doc = re.sub(r'(?!\D)\d{2,}x\d{2,}(?=\D)', ' PlHolderResolution ', doc)

            if self.args.stem:
                if self.args.stop:
                    doc = [PorterStemmer().stem(t) for t in self.tokeniser(doc) if t not in stopwords.words('english')]
                else:
                    doc = [PorterStemmer().stem(t) for t in self.tokeniser(doc)]
            else:
                if self.args.stop:
                    doc = [t for t in self.tokeniser(doc) if t not in stopwords.words('english')]
                else:
                    doc = list(self.tokeniser(doc))
            if len(doc):
                print(' '.join(doc), file=self.txt, end=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse HTML')
    parser.add_argument('-s', '--stop', action='store_true', required=False, help='Stop')
    parser.add_argument('-e', '--stem', action='store_true', required=False, help='Stem')
    parser.add_argument('-m', '--mime', action='store_true', required=False, help='Remove MIME Header')
    parser.add_argument('-d', '--digit', action='store_true', required=False, help='Substitute Digits')
    parser.add_argument('-o', '--other', action='store_true', required=False, help='Other preprocessing')

    args = parser.parse_args()
    print(args)
    # for arg in vars(args):
    #     if vars(args)[arg] == 'True':
    #         vars(args)[arg] = True
    #     elif vars(args)[arg] == 'False':
    #         vars(args)[arg] = False
    #     else:
    #         raise
    root_dir = 'tokens' + getRootSuffix(args)
    print('Stored to {:s}'.format(root_dit))
    os.makedirs(root_dir)

    allfiles = glob.glob('webkb/**/*', recursive=True)
    
    for file in tqdm(allfiles):
        try:
            html = open(file).read()
        except UnicodeDecodeError:
            html = open(file, encoding='iso-8859-1').read()
        except IsADirectoryError:
            continue
        except:
            print(file)
            raise

        label, uni, name = file.strip().split('/')[1:]
        os.makedirs(os.path.join(root_dir, label, uni), exist_ok=True)
        with open(os.path.join(root_dir, label, uni, name+'.txt'), 'a+', encoding='utf-8') as txt:
            parser = MyHTMLParser(txt, args)
            parser.feed(html)