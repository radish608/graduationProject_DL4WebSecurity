# encoding: utf-8

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import *
from sklearn import svm, ensemble, feature_selection

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
import PreProcess
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np


class Classifier(object):
    def __init__(self, classifier=LogisticRegression):
        self.classifier = classifier
        self.clf = None

    def train_func(self, x_data, y_data):
        train_model = self.classifier(class_weight='balanced')
        print x_data.shape
        self.clf = train_model.fit(x_data, y_data)
        return train_model

def result_vectoring(v):
    v = v.reshape(-1, 1)
    return np.concatenate((v*(-1)+1, v), axis=1)


def feature_select(X, y):
    return SelectKBest(chi2, k=500).fit_transform(X, y)


if __name__ == "__main__":
    ham = './ham_all.txt'
    spam = './spam_all.txt'
    models = [LogisticRegression, ensemble.RandomForestClassifier, svm.LinearSVC]

    pre_process = 0
    is_test = 1
    if pre_process:
        PreProcess.process_data([ham, spam])  # 处理原始数据，tf-idf训练后并保存
    if is_test:
        x, y, model = PreProcess.load_data()  # 加载tf-idf数据
        x = feature_select(x, y)
        #print len(model.get_feature_names())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)
        model = Classifier()
        model.train_func(x_train, y_train)
        y_pred = model.clf.predict(x_test)
        y_vec = result_vectoring(y_test)
        y_score = model.clf.predict_proba(x_test)
        print 'roc_auc_score:', metrics.roc_auc_score(y_vec, y_score)
        print 'classification_report\n', metrics.classification_report(y_test, y_pred)
