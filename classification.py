from sklearn import svm, metrics
import scipy.sparse as sp
import numpy as np
import generateverctor


def splitvector(vectors, labels, uni, testuni):
    # split vector and label to train and test
    idx = [i for i, x in enumerate(uni) if x == testuni]
    test_vector = vectors[idx]
    test_label = np.array(labels)[idx]
    train_vector = sp.csr_matrix(np.delete(vectors.toarray(), idx, 0))
    train_label = np.delete(labels, idx)
    return train_vector, train_label, test_vector, test_label


def svmclassfier(train_vector, train_label, test_vector):
    lin_clf = svm.LinearSVC()
    lin_clf.fit(train_vector, train_label)
    predict = lin_clf.predict(test_vector)
    return predict


if __name__ == '__main__':
    vectors, labels, uni, features = generateverctor.tfidf()
    train_vector, train_label, test_vector, test_label = splitvector(vectors, labels, uni, "cornell")
    predict = svmclassfier(train_vector, train_label, test_vector)
    # print(vectors.shape, train_vector.shape, len(train_label), test_vector.shape, len(test_label))
    #print(predict[0:10])
    #print(test_label[0:10])
    print(metrics.precision_score(test_label, predict, average='weighted'))

