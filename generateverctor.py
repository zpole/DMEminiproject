from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

def readfile(filepath):
    contants = []
    labels = []
    with open(filepath) as fp:
        for line in fp.readlines():
            line = "processed/" + line.strip() + ".txt"
            #read contants of each file
            try:
                with open(line) as file:
                    data = file.read().replace("\n", " ")
                    contants.append(data)
            except:
                continue
            # add label for each file
            tags = line.split("/")
            if tags[1] == 'course':
                labels.append('0')
            elif tags[1] == 'department':
                labels.append('1')
            elif tags[1] == 'faculty':
                labels.append('2')
            elif tags[1] == 'other':
                labels.append('3')
            elif tags[1] == 'project':
                labels.append('4')
            elif tags[1] == 'staff':
                labels.append('5')
            elif tags[1] == 'student':
                labels.append('6')
    return contants, labels

def tfidf(filepath):
    contants, labels = readfile(filepath)
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(contants)
    return vectors, labels, tfidf.get_feature_names()

def bow(filepath):
    contants, labels = readfile(filepath)
    bow = CountVectorizer(stop_words="english")
    vectors = bow.fit_transform(contants)
    return vectors, labels, bow.get_feature_names()

# test
if __name__ == '__main__':
    # contants, labels = readfile("testpath.txt")
    # print(contants[1], labels[1])
    vectors, labels, features = tfidf("allfiles.txt")
    print(vectors[1])
