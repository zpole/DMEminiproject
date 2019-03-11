from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import glob


def readfile():
    contants = []
    labels = []
    uni = []
    allfiles = glob.glob('processed/**/*', recursive=True)
    for line in allfiles:
        try:
            with open(os.path.abspath(line)) as file:
                data = file.read().replace("\n", " ")
                contants.append(data)
        except:
            continue
        tags = os.path.normpath(line).split(os.sep)
        # add label for each file
        labels.append(tags[1])
        # add university for each file
        uni.append(tags[2])
    return contants, labels, uni


def tfidf():
    contants, labels, uni = readfile()
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(contants)
    return vectors, labels, uni, tfidf.get_feature_names()  # uni = university


def bow():
    contants, labels, uni = readfile()
    bow = CountVectorizer(stop_words="english")
    vectors = bow.fit_transform(contants)
    return vectors, labels, uni, bow.get_feature_names()


# test
if __name__ == '__main__':
    # contants, labels = readfile("testpath.txt")
    # print(contants[1], labels[1])
    vectors, labels, uni, features = tfidf()
    print(vectors.shape, len(labels), len(uni))
