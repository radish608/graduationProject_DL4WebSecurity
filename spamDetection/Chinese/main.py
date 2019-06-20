from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn
from sklearn.feature_selection import *
from sklearn import ensemble, feature_selection
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
from sklearn import preprocessing
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import preprocess
import joblib

max_features=5000

def feature_select(X, y):
    return SelectKBest(chi2, k=500).fit_transform(X, y)

def do_metrics(y_test,y_pred):
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


class CNNTrainModel():
    def __init__(self, x, y):
        self.max_document_length=500
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
        self.x_train = pad_sequences(x_train, maxlen=self.max_document_length, value=0.)
        self.x_test = pad_sequences(x_test, maxlen=self.max_document_length, value=0.)
        self.y_train = to_categorical(y_train, nb_classes=2)
        self.y_test = to_categorical(y_test, nb_classes=2)
        self.feature_name = "vocabulary table"
        self.model_path = "./Model/CNN/cnn_{}".format(self.feature_name)
        print type(self.y_test)

    def cnn(self):
        network = input_data(shape=[None, self.max_document_length], name='input')
        network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
        branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
        branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
        branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
        network = merge([branch1, branch2, branch3], mode='concat', axis=1)
        network = tf.expand_dims(network, 2)
        network = global_max_pool(network)
        network = dropout(network, 0.8)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy', name='target')

        model = tflearn.DNN(network, tensorboard_verbose=0)
        return model

    def do_cnn(self):
        model = self.cnn()
        model.fit(self.x_train, self.y_train,
                  n_epoch=5, shuffle=True, validation_set=(self.x_test, self.y_test),
                  show_metric=True, batch_size=100,run_id="spam_cnn_{}".format(self.feature_name))
        model.save(self.model_path)

    def show_result(self):
        model = self.cnn()
        model.load(self.model_path)
        #y_pred = model.predict(self.x_test)
        #do_metrics(self.y_test, y_pred)
        #print model.evaluate(self.x_test, self.y_test)
        y_predict_list=model.predict(self.x_test)
        #y_predict = list(model.predict(self.x_test, as_iterable=True))
        y_predict=[]
        for i in y_predict_list:
            if i[0] > 0.5:
                y_predict.append(0)
            else:
                y_predict.append(1)

        y_test_list = []
        for i in self.y_test:
            y_test_list.append(i[1])

        do_metrics(y_test_list, y_predict)


class MLPTrainModel():
    def __init__(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
        self.feature_name = "tfidf"
        self.model_path = "./Model/MLP/mlp_{}".format(self.feature_name)

    def mlp(self):
        clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            hidden_layer_sizes = (5, 2),
                            random_state = 1)
        return clf

    def do_mlp(self):
        clf = self.mlp()
        clf.fit(self.x_train, self.y_train)
        joblib.dump(clf, self.model_path)
        y_pred = clf.predict(self.x_test)

    def show_result(self):
        clf = joblib.load(self.model_path)
        y_pred = clf.predict(self.x_test)
        print clf
        do_metrics(self.y_test, y_pred)


class RNNTrainModel():
    def __init__(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
        self.x_train = pad_sequences(x_train, maxlen=500, value=0.)
        self.x_test = pad_sequences(x_test, maxlen=500, value=0.)
        self.y_train = to_categorical(y_train, nb_classes=2)
        self.y_test = to_categorical(y_test, nb_classes=2)
        #self.feature_name = "tfidf"
        self.model_path = "./Model/RNN/rnn"

    def rnn(self):
        net = tflearn.input_data([None, 500])
        net = tflearn.embedding(net, input_dim=100000, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.1,
                                 loss='categorical_crossentropy')

        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model

    def do_rnn(self):
        model = self.rnn()
        model.fit(self.x_train, self.y_train, validation_set=(self.x_test, self.y_test), show_metric=True,
                  batch_size=10,run_id="spam-run",n_epoch=5)
        model.save(self.model_path)


if __name__ == "__main__":
    #show_diffrent_max_features()

    """
    MLP
    """
    x, y = preprocess.get_features_by_tfidf()
    #mlp_tm = MLPTrainModel(x, y)
    #mlp_tm.do_mlp()
    #mlp_tm.show_result()

    """
    CNN
    """
    #x, y = preprocess.get_features_by_vt()
    #cnn_tm = CNNTrainModel(x, y)
    #cnn_tm.do_cnn()
    #cnn_tm.show_result()

    """
    RNN
    """
    #x, y = preprocess.get_features_by_vt()
    rnn_tm = RNNTrainModel(x, y)
    rnn_tm.do_rnn()
