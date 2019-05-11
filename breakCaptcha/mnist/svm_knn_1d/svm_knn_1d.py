# -*- coding: utf-8 -*-
import tflearn
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn.externals import joblib
import tflearn.datasets.mnist as mnist

def evaluate(x_test, y_test, modelpath):
    clf = joblib.load(modelpath)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def svm_1d(x_train, y_train,x_test, y_test, flag):
    print "SVM + 1d"
    if flag==0:
        clf = svm.SVC(decision_function_shape='ovo')
        print clf
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print metrics.accuracy_score(y_test, y_pred)
        print metrics.confusion_matrix(y_test, y_pred)
        joblib.dump(clf, "svm_1d.m")
    elif flag==1:
        evaluate(x_test, y_test, "svm_1d.m")

def knn_1d(x_train, y_train,x_test, y_test, flag):
    print "KNN + 1d"
    if flag==0:
        clf = neighbors.KNeighborsClassifier(n_neighbors=15)
        print clf
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print metrics.accuracy_score(y_test, y_pred)
        print metrics.confusion_matrix(y_test, y_pred)
        joblib.dump(clf, "knn_1d.m")
    elif flag==1:
        evaluate(x_test, y_test, "knn_1d.m")

if __name__ == '__main__':
    X, Y, testX, testY = mnist.load_data(one_hot=False)
    flag = 1
    svm_1d(X, Y, testX, testY, flag)
    knn_1d(X, Y, testX, testY, flag)
