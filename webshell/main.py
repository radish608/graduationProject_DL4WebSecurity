# -*-coding: utf-8 -*-
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import commands
import pickle
import joblib
from sklearn.metrics import classification_report
import preprocess

max_features=10000
min_opcode_count=2

#pro
#webshell_dir="../Datasets/dataset_webshell/b/"
#whitefile_dir="../Datasets/dataset_webshell/w/"
webshell_dir="../Datasets/dataset_webshell/webshell/PHP/"
whitefile_dir="../Datasets/dataset_webshell/normal/php/"
check_dir="~/Downloads/php-exploit-scripts/"
white_count=0
black_count=0
php_bin="/usr/bin/php"

def show_diffrent_max_features():
    global max_features
    a=[]
    b=[]
    for i in range(1000,20000,2000):
        max_features=i
        print "max_features=%d" % i
        x, y = preprocess.get_features_by_tf()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        #gnb = GaussianNB()
        #gnb.fit(x_train, y_train)
        #y_pred = gnb.predict(x_test)
        score=metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()


def check_webshell(clf,dir):
    all=0
    all_php=0
    webshell=0

    webshell_files_list = load_files_re(webshell_dir)
    CV = CountVectorizer(ngram_range=(3, 3), decode_error="ignore", max_features=max_features,
                         token_pattern=r'\b\w+\b', min_df=1, max_df=1.0)
    x = CV.fit_transform(webshell_files_list).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit_transform(x)


    g = os.walk(dir)
    for path, d, filelist in g:
        for filename in filelist:
            fulepath=os.path.join(path, filename)
            t = load_file(fulepath)
            t_list=[]
            t_list.append(t)
            x2 = CV.transform(t_list).toarray()
            x2 = transformer.transform(x2).toarray()
            y_pred = clf.predict(x2)
            all+=1
            if filename.endswith('.php'):
                all_php+=1
            if y_pred[0] == 1:
                print "%s is webshell" % fulepath
                webshell+=1

    print "Scan %d files(%d php files),%d files is webshell" %(all,all_php,webshell)


def do_check(x,y,clf):
    clf.fit(x, y)
    print "check_webshell"
    check_webshell(clf,check_dir)

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

class MLPTrainModel():
    def __init__(self, x, y, feature_name):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        self.feature_name = feature_name

    def mlp(self):
        clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-5,
                            hidden_layer_sizes=(7, 7),
                            random_state=1,
                            learning_rate_init=0.001)
        return clf

    def do_mlp(self):
        clf = self.mlp()
        model_path = "./Model/MLP/MLP_{}".format(self.feature_name)
        clf.fit(self.x_train, self.y_train)
        print clf
        joblib.dump(clf, model_path)

    def show_result(self):
        clf = joblib.load(model_path)
        y_pred = clf.predict(self.x_test)
        do_metrics(self.y_test, y_pred)


class CNNTrainModel():
    def __init__(self, x, y, feature_name):
        self.max_document_length = 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        self.x_train = pad_sequences(x_train, maxlen=self.max_document_length, value=0.)
        self.x_test = pad_sequences(x_test, maxlen=self.max_document_length, value=0.)
        self.y_train = to_categorical(y_train, nb_classes=2)
        self.y_test = to_categorical(y_test, nb_classes=2)
        self.feature_name = feature_name

    def cnn(self):
        network = input_data(shape=[None,self.max_document_length], name='input')
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
        model_name = "./Model/CNN/{}/CNN_{}".format(self.feature_name, self.feature_name)
        model.fit(self.x_train, self.y_train,
                    n_epoch=10, shuffle=True, validation_set=0.1,
                    show_metric=True, batch_size=100,run_id="webshell")
        model.save(model_name)

    def show_result(self):
        model = self.cnn()
        model_name = "./Model/CNN/{}/CNN_{}".format(self.feature_name, self.feature_name)
        print model_name
        model.load(model_name)
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


class RNNTrainModel():
    def __init__(self, x, y, feature_name):
        self.max_document_length = 100
        self.model_name = "./Model/RNN/RNN_{}".format(feature_name)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        self.x_train = pad_sequences(x_train, maxlen=max_document_length, value=0.)
        self.x_test = pad_sequences(x_test, maxlen=max_document_length, value=0.)
        self.y_train = to_categorical(y_train, nb_classes=2)
        self.y_test = to_categorical(y_test, nb_classes=2)

    def rnn(self):
        net = tflearn.input_data([None, max_document_length])
        net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        return model

    def do_rnn(self):
        model = self.rnn()
        model.fit(self.x_train, self.y_train, validation_set=0.1, show_metric=True,
                    batch_size=10,run_id="webshell",n_epoch=5)
        model.save(model_name)

    def show_result(self):
        model = self.rnn()
        model.load(self.model_name)
        y_predict_list=model.predict(self.x_test)
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

if __name__ == '__main__':

    """
    MLP
    """
    #x, y = preprocess.get_feature_by_ngram()
    #feature_name="ngram"

    #x, y = preprocess.get_feature_by_opcode_ngram()
    #feature_name = "opcode_ngram"

    #mlp_tm = MLPTrainModel(x, y, feature_name)
    #mlp_tm.do_mlp()
    #mlp_tm.show_result() #ok: opcode_ngram, php_ngram



    """
    CNN
    """
    #x, y = preprocess.get_feature_by_opcode_vt()
    #feature_name = "opcode_vt"

    x, y = preprocess.get_feature_by_vt()
    feature_name = "php_vt"

    cnn_tm = CNNTrainModel(x, y, feature_name)
    #cnn_tm.do_cnn()
    cnn_tm.show_result() #ok:opcode_vt & php_vt
