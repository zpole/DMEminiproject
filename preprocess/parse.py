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
        self.tokeniser = CountVectorizer(token_pattern=r'(?u)\b\w\w+\b').build_tokenizer()

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
            doc = ' ' + doc + ' '

            if self.args.other:
                doc = re.sub(r'\bph\.? ?d\.?\b', 'phd', doc)
            
            if self.args.digit:
                doc = re.sub(r'(\D)\d{1,2}:\d\d:\d\d(\D)', r'\1 PlHolderTime \2', doc)
                doc = re.sub(r'(\D)\d{1,2}:\d\d(\D)', r'\1 PlHolderTime \2', doc)

                doc = re.sub(r'(\D)[\dx]{3}\-[\dx]{4}(\D)', r'\1 PlHolderPhoneNum \2', doc)
                doc = re.sub(r'(\D)[\dx]{3}\D[\dx]{3}\-[\dx]{4}(\D)', r'\1 PlHolderPhoneNum \2', doc)
                doc = re.sub(r'(\D\D)[\dx]{3}\D\D[\dx]{3}\-[\dx]{4}(\D)', r'\1 PlHolderPhoneNum \2', doc)
                
                doc = re.sub(r'(\D)\d{5}-\d{4}(\D)', r'\1 PlHolderZipPlusFour \2', doc)
                
                doc = re.sub(r'(\D)\d{2,}x\d{2,}x\d{2,}(\D)', r'\1 PlHolderResolution \2', doc)
                doc = re.sub(r'(\D)\d{2,}x\d{2,}(\D)', r'\1 PlHolderResolution \2', doc)

                doc = re.sub(r'(\D)[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}(\D)', r'\1 PlHolderUUID \2', doc)
                
                # 0 10000101 00000000110011001100110
                doc = re.sub(r'(\D)[01]  ?[01]{8}  ?[01]{23}(\D)', r'\1 PlHolderBinaryThirtyTwo \2', doc)

                # 00000000000000000000000000011001
                doc = re.sub(r'(\D)[01]{32}(\D)', r'\1 PlHolderBinThirtyTwoBit \2', doc)

                # 10000101
                doc = re.sub(r'(\D)[01]{8}(\D)', r'\1 PlHolderBinEightBit \2', doc)
                
                # 0000200 4865 6c6c 6f2c 2057 6f72 6c64 0a00 4865
                doc = re.sub(r'(\D)[\da-f]{7}( [\da-f]{4})*(\D)', r'\1 PlHolderBinFile \3', doc)

                # 0100 0010 1000 0000 0110 0110 0110 0110
                doc = re.sub(r'(\D)[01]{4}( [01]{4}){7}(\D)', r'\1 PlHolderThirtyTwoBitHexInBin \3', doc)

                # 0100 0010
                doc = re.sub(r'(\D)[01]{4} [01]{4}(\D)', r'\1 PlHolderEightBitHexInBin \2', doc)

                # 0x4288f873a
                doc = re.sub(r'(\D)0x[\da-f]{8}(\D)', r'\1 PlHolderThirtyTwoBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{8}(\D)', r'\1 PlHolderTwentyEightBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{6}(\D)', r'\1 PlHolderTwentyFourBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{6}(\D)', r'\1 PlHolderTwentyBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{4}(\D)', r'\1 PlHolderSixteenBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{3}(\D)', r'\1 PlHolderTwelveBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f]{2}(\D)', r'\1 PlHolderEightBitHex \2', doc)
                doc = re.sub(r'(\D)0x[\da-f](\D)', r'\1 PlHolderFourBitHex \2', doc)

                doc = re.sub(r'(?!\D)\d(?=\D)', ' PlHolderOneDigit ', doc)
                doc = re.sub(r'(?!\D)\d{2}(?=\D)', ' PlHolderTwoDigit ', doc)
                doc = re.sub(r'(?!\D)\d{3}(?=\D)', ' PlHolderThreeDigit ', doc)
                doc = re.sub(r'(?!\D)\d{4}(?=\D)', ' PlHolderFourDigit ', doc)
                doc = re.sub(r'(?!\D)\d{5}(?=\D)', ' PlHolderFiveDigit ', doc)
                doc = re.sub(r'(?!\D)\d{6}(?=\D)', ' PlHolderSixDigit ', doc)
                doc = re.sub(r'(?!\D)\d{7}(?=\D)', ' PlHolderSevenDigit ', doc)
                doc = re.sub(r'(?!\D)\d{8}(?=\D)', ' PlHolderEightDigit ', doc)
                doc = re.sub(r'(?!\D)\d+(?=\D)', ' PlHolderDigits ', doc)
                
            doc = re.sub(r'_+', ' ', doc)

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
    print('Stored to {:s}'.format(root_dir))
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

        partition, label, uni, name = file.strip().split('/')[1:]
        os.makedirs(os.path.join(root_dir, partition, label, uni), exist_ok=True)
        with open(os.path.join(root_dir, partition, label, uni, name+'.txt'), 'a+', encoding='utf-8') as txt:
            parser = MyHTMLParser(txt, args)
            parser.feed(html)