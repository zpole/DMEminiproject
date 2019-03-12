from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import glob


def readfile():
    contents = []
    labels = []
    uni = []  # university index
    filename = []
    allfiles = glob.glob('processed/**/*', recursive=True)
    for line in allfiles:
        try:
            with open(os.path.abspath(line)) as file:
                data = file.read().replace("\n", " ")
                contents.append(data)
        except:
            continue
        tags = os.path.normpath(line).split(os.sep)
        # add label for each file
        labels.append(tags[1])
        # add university for each file
        uni.append(tags[2])
        filename.append(tags[3][:-4])
    return contents, labels, uni, filename


def tfidf():
    contents, labels, uni, filename = readfile()
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(contents)
    return vectors, labels, uni, filename, tfidf.get_feature_names()  # uni = university


def bow():
    contents, labels, uni = readfile()
    bow = CountVectorizer(stop_words="english")
    vectors = bow.fit_transform(contents)
    return vectors, labels, uni, filename, bow.get_feature_names()


# test
if __name__ == '__main__':
    # contents, labels = readfile()
    # print(contents[1], labels[1])
    vectors, labels, uni, filename, features = tfidf()
    print(vectors.shape, len(labels), len(uni))
