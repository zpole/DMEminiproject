from sklearn import svm
import scipy.sparse as sp
import numpy as np
import generateverctor

def splitvector(vectors, labels, uni, testuni):
    idx = [i for i,x in enumerate(uni) if x == testuni]
    test_vector = vectors[idx]
    test_label = np.array(labels)[idx]
    train_vector = sp.csr_matrix(np.delete(vectors.toarray(), idx, 0))
    train_label = np.delete(labels, idx)
    return train_vector, train_label, test_vector, test_label

def svmclassfier(vectors, labels, testuni):

    lin_clf = svm.LinearSVC()
    lin_clf.fit(vectors, labels)

if __name__ == '__main__':
    vectors, labels, uni, features = generateverctor.tfidf("allfiles.txt")
    train_vector, train_label, test_vector, test_label = splitvector(vectors, labels, uni, "cornell")
    print(vectors.shape, train_vector.shape, len(train_label), test_vector.shape, len(test_label))
