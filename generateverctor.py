from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import glob
from preprocess.textprepro import create_vectoriser
from preprocess.parse import getRootDir


def readfile(stem, stop):
    contents = []
    labels = []
    uni = []  # university index
    filename = []

    root_dir = getRootDir(stem=stem, stop=stop)

    allfiles = glob.glob(root_dir + '/**/*', recursive=True)
    for line in allfiles:
        try:
            file = open(os.path.abspath(line))
        except IsADirectoryError:
            continue
        except:
            raise

        # contents.append('')
        # for ln in file:
        #     data = txtpp.pp(ln)
        #     contents[-1] += ' ' + data

        contents.append(file.read())

        tags = os.path.normpath(line).split(os.sep)
        # add label for each file
        labels.append(tags[1])
        # if tags[1] == 'course':
        #     labels.append(0)
        # elif tags[1] == 'department':
        #     labels.append(1)
        # elif tags[1] == 'faculty':
        #     labels.append(2)
        # elif tags[1] == 'other':
        #     labels.append(3)
        # elif tags[1] == 'project':
        #     labels.append(4)
        # elif tags[1] == 'staff':
        #     labels.append(5)
        # elif tags[1] == 'student':
        #     labels.append(6)
        # add university for each file
        uni.append(tags[2])
        filename.append(tags[3][:-4])
    return contents, labels, uni, filename


def vectoriser(vec, stem=False, stop=True):
    contents, labels, uni, filename = readfile(stem=stem, stop=stop)
    V = create_vectoriser(vec)
    vectors = V.fit_transform(contents)
    return vectors, labels, uni, filename, V.get_feature_names()  # uni = university


# def tfidf():
#     contents, labels, uni, filename = readfile()
#     V = create_vectoriser('tfidf', stem=False, stop=True)
#     vectors = V.fit_transform(contents)
#     return vectors, labels, uni, filename, V.get_feature_names()  # uni = university


# def bow():
#     contents, labels, uni, filename = readfile()
#     V = create_vectoriser('count', stem=False, stop=True)
#     vectors = V.fit_transform(contents)
#     return vectors, labels, uni, filename, V.get_feature_names()


# test
if __name__ == '__main__':
    # contents, labels = readfile()
    # print(contents[1], labels[1])
    vectors, labels, uni, filename, features = vectoriser('tfidf', stem=False, stop=True)
    print(vectors.shape, len(labels), len(uni))
