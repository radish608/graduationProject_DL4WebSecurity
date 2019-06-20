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
import matplotlib.pyplot as plt


class TrainModel():
    def __init__(self, n, m=0):
        self.n = n
        self.m = m
        if self.m!= 0:
            self.model_path = "./Model/linuxBackdoor_{}+{}-Gram.m".format(self.n, self.m)
            self.x_train, self.x_test, self.y_train, self.y_test = preprocess.get_feature_wordbag(self.n, self.m)
        else:
            self.model_path = "./Model/linuxBackdoor_{}-Gram.m".format(self.n)
            self.x_train, self.x_test, self.y_train, self.y_test = preprocess.get_feature_wordbag(self.n)


    def do_mlp(self):
        clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            hidden_layer_sizes = (5, 2),
                            random_state = 1)
        clf.fit(self.x_train, self.y_train)
        joblib.dump(clf, self.model_path)
        y_pred = clf.predict(self.x_test)
        print(classification_report(self.y_test, y_pred))
        print metrics.confusion_matrix(self.y_test, y_pred)

    def show_result(self):
        clf = joblib.load(self.model_path)
        y_pred = clf.predict(self.x_test)
        #print(classification_report(self.y_test, y_pred))
        self.do_metrics(self.y_test, y_pred)

    def do_metrics(self, y_test,y_pred):
        print "metrics.confusion_matrix:"
        print metrics.confusion_matrix(y_test, y_pred)
        print "metrics.accuracy_score:"
        print metrics.accuracy_score(y_test, y_pred)
        print "metrics.precision_score:"
        print metrics.precision_score(y_test, y_pred)
        print "metrics.recall_score:"
        print metrics.recall_score(y_test, y_pred)
        print "metrics.f1_score:"
        print metrics.f1_score(y_test,y_pred)


if __name__ == "__main__":
    tm = TrainModel(2, 3)
    #tmc.do_mlp()
    tm.show_result()
