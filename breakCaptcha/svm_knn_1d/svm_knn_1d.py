# -*- coding: utf-8 -*-
import tflearn
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn.externals import joblib
import tflearn.datasets.mnist as mnist

def svm_1d(x_train, y_train,x_test, y_test):
    print "SVM + 1d"
    clf = svm.SVC(decision_function_shape='ovo')
    print clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    joblib.dump(clf, "svm_1d.m")

def knn_1d(x_train, y_train,x_test, y_test):
    print "KNN + 1d"
    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    print clf
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    joblib.dump(clf, "knn_1d.m")

if __name__ == '__main__':
    X, Y, testX, testY = mnist.load_data(one_hot=False)
    svm_1d(X, Y, testX, testY)
    knn_1d(X, Y, testX, testY)
