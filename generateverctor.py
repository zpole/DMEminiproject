from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from os import path

def readfile(filepath):
    contants = []
    labels = []
    uni = []
    with open(path.abspath(filepath)) as fp:
        for line in fp.readlines():
            line = "processed/" + line.strip() + ".txt"
            #read contants of each file
            try:
                with open(path.abspath(line)) as file:
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
            # add university for each file
            if tags[2] == 'cornell':
                uni.append('cornell')
            elif tags[2] == 'misc':
                uni.append('misc')
            elif tags[2] == "texas":
                uni.append("texas")
            elif tags[2] == "washington":
                uni.append("washington")
            elif tags[2] == "wisconsin":
                uni.append("wisconsin")
    return contants, labels, uni

def tfidf(filepath):
    contants, labels, uni = readfile(filepath)
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(contants)
    return vectors, labels, uni, tfidf.get_feature_names()  # uni = university

def bow(filepath):
    contants, labels, uni = readfile(filepath)
    bow = CountVectorizer(stop_words="english")
    vectors = bow.fit_transform(contants)
    return vectors, labels, uni, bow.get_feature_names()

# test
if __name__ == '__main__':
    # contants, labels = readfile("testpath.txt")
    # print(contants[1], labels[1])
    vectors, labels, uni, features = tfidf("allfiles.txt")
    print(vectors.shape, len(labels), len(uni))
