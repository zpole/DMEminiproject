from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import generatevector
from classification import splitvector
import time
import argparse
from sklearn.linear_model import LogisticRegression


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.array(classes))
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def EvaluationModel(test_label, predict, classes, cfsm = False):
    print('Accuracy: {:.6f}'.format(accuracy_score(test_label, predict)))
    print('Precision:{:.6f}'.format(precision_score(test_label, predict, average='weighted')))
    print('Recall:{:.6f}'.format(recall_score(test_label, predict, average='weighted')))
    print('F1_Score:{:.6f}'.format(f1_score(test_label, predict, average='weighted')))

    if cfsm:
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(test_label, predict, classes=classes,
                            title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plot_confusion_matrix(test_label, predict, classes=classes, normalize=True,
                            title='Normalized confusion matrix')
        plt.show()

def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    return s, m, h


def CrossValidation(model, vectors, labels, uni, classes):
    universities = ['cornell', 'texas', 'washington', 'wisconsin']
    label_t = []
    label_p = []
    for university in universities:
        start = time.time()
        train_vector, train_label, test_vector, test_label = splitvector(vectors, labels, uni, university)
        model.fit(train_vector, train_label)
        predict = model.predict(test_vector)
        # score = model.score(test_vector, test_label)
        label_t = np.append(label_t,list(test_label))
        label_p = np.append(label_p,list(predict))
        end = time.time()
        s, m, h = exec_time(start,end)
        print('\nFinished model {} on {} validation set'.format(type(model).__name__, university))
        print('Execution Time: ' + '{0:02d}:{1:02d}:{2:02d}'.format(h, m, s))
        print('Classification performance on {} set:'.format('university'))
        EvaluationModel(test_label, predict, classes)
    print('='*30 + ' Report overall cross validation performance ' + '=' * 30)
    EvaluationModel(label_t, label_p, classes, cfsm = True)
    return label_t, label_p

    # print('CrossValidation Performance: ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse HTML')
    parser.add_argument('-s', '--stop', action='store_true', required=False, help='Stop')
    parser.add_argument('-e', '--stem', action='store_true', required=False, help='Stem')
    parser.add_argument('-m', '--mime', action='store_true', required=False, help='Remove MIME Header')
    parser.add_argument('-d', '--digit', action='store_true', required=False, help='Substitute Digits')
    parser.add_argument('-o', '--other', action='store_true', required=False, help='Other preprocessing')
    args = parser.parse_args()

    vectors, labels, uni, filename, features = generatevector.vectoriser('tfidf', args)
    classes = ["course", "department", "faculty", "other", "project", "staff", "student"]

    lr_clf = LogisticRegression()
    label_t, label_p = CrossValidation(lr_clf, vectors, labels, uni, classes)
    