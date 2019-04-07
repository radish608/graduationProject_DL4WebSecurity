# -*-coding: utf-8 -*-

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
import tensorflow as tf

#加载数据
import tflearn.datasets.mnist as mnist


#k邻近算法+1d
def knn_1d(x_train, y_train, x_test, y_test):
    print "Knn + 1d"
    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    print clf
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_train)
    print metrics.accuracy_score(y_test, y_predict)

#多层感知机+1d
def mlp_1d(x_train, y_train,x_test , y_test):
    print "MLP + 1d"

    #构造神经网络
    input_layer = tflearn.input_data(shape=[None, 784])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')


    #训练
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
                             show_metric=True, run_id="mnist")
    model_path='bc.tfl'
    model.save(model_path)

if __name__ == '__main__':
    print "break CAPTCHA"
    #一维向量
    X, Y, testX, testY = mnist.load_data(one_hot=True)

    #二维向量
    #X = X.reshape([-1, 28, 28, 1])
    #testX = testX.reshape([-1, 28, 28, 1])
    print testX
    #knn_1d(X, Y, testX,testY)
    mlp_1d(X, Y, testX,testY)
