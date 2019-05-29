# -*-coding: utf-8 -*-

from sklearn import metrics
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
import tensorflow as tf
import preprocess

#多层感知机+1d
def do_mlp():

    #构造神经网络
    input_layer = tflearn.input_data(shape=[None, 100, 60, 1])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 252, activation='softmax')

    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')


    #训练
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model_path='./model/mlp/mlp'
    flag = 0
    if flag == 0:
        if os.path.exists('./model/mlp'):
            print "loading model"
            model.load(model_path)
        #for i in range(10):
        x, y = preprocess.get_feature()
        x = x.reshape([-1, 100, 60, 1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
        model.fit(x_train, y_train, n_epoch=100000, validation_set=(x_test, y_test),
                              show_metric=True, run_id="captchalib")
        model.save(model_path)
    elif flag==1:
        model.load(model_path)
        print model.evaluate(x_test, y_test)

if __name__ == '__main__':
    #一维向量
    #X, Y, testX, testY = mnist.load_data(one_hot=True)

    #二维向量
    #X = X.reshape([-1, 28, 28, 1])
    #testX = testX.reshape([-1, 28, 28, 1])
    #print testX
    do_mlp()
