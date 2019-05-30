# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import re
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import preprocess

model_path = "./Model/linuxBackdoor_{}-Gram.m"

def do_mlp(x_train, x_test, y_train, y_test, n):
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x_train, y_train)
    #joblib.dump(clf, model_path.format(n))
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def show_result(x_train, x_test, y_train, y_test, n):
    clf = joblib.load(model_path.format(n))
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    #for i in range(2, 6):
        #print "{}-Gram&tf-idf and mlp".format(i)
        #x_train, x_test, y_train, y_test = preprocess.get_feature_wordbag(i, i)
        #do_mlp(x_train, x_test, y_train, y_test, i)
        #show_result(x_train, x_test, y_train, y_test, i)
    x_train, x_test, y_train, y_test = preprocess.get_feature_wordbag(3, 4)
    do_mlp(x_train, x_test, y_train, y_test, 2)
