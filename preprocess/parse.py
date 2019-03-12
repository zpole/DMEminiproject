from html.parser import HTMLParser
import glob
import re
import os
from tqdm import tqdm
import unidecode

allfiles = glob.glob('webkb/**/*', recursive=True)

class MyHTMLParser(HTMLParser):
    def __init__(self, txt):
        HTMLParser.__init__(self)
        self.txt = txt

    def handle_starttag(self, tag, attrs):
        pass
        # print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        pass
        # print("Encountered an end tag :", tag)

    def handle_data(self, data):
        data = re.sub(r'[\n\t ]+', ' ', data.strip())
        data = unidecode.unidecode(data)
        if data is not '':
            print(data.strip(), file=self.txt)


# class MyHTMLParser(HTMLParser):
#     def __init__(self):
#         HTMLParser.__init__(self)
#         self.level = ['root']
#         self.d = dict()
#         self.d['root'] = {'TextContent':''}
        
#     def handle_starttag(self, tag, attrs):
#         i = 0
#         tmp_d = self.d
#         while i < len(self.level):
#             tmp_d = tmp_d[self.level[i]]
#             i += 1
#         if tag not in tmp_d: 
#             tmp_d[tag] = {'TextContent':''}
#         self.level.append(tag)
#         # print('\"{:s}\":{{'.format(tag))
#         # print("Encountered a start tag:", tag)

#     def handle_endtag(self, tag):
#         while self.level[-1] != tag:
#             self.level.pop(-1)
#         if self.level[-1] == tag:
#             self.level.pop(-1)
#         else:
#             raise()
#         # print('}')
#         # print("Encountered an end tag :", tag)

#     def handle_data(self, data):
#         i = 0
#         tmp_d = self.d
#         while i < len(self.level):
#             tmp_d = tmp_d[self.level[i]]
#             i += 1
#         data = re.sub(r'[\n\t ]+', ' ', data.strip())
#         tmp_d['TextContent'] += ' ' + data
#         # if data is not '':
#             # print('\"TextContent\":\"{:s}\"'.format(data))



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
    os.makedirs(os.path.join('processed', label, uni), exist_ok=True)
    with open(os.path.join('processed', label, uni, name+'.txt'), 'a+', encoding='utf-8') as txt:
        parser = MyHTMLParser(txt)
        parser.feed(html)