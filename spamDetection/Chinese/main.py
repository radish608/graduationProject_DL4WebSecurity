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
max_document_length=100

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

def show_result(x_train, x_test, y_train, y_test, n):
    clf = joblib.load(model_path.format(n))
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print metrics.confusion_matrix(y_test, y_pred)

def do_cnn(trainX, testX, trainY, testY, feature_name):
    global max_document_length

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)

    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    model_path = "./Model/CNN/cnn_{}".format(feature_name)

    network = input_data(shape=[None,max_document_length], name='input')
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
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="spam_cnn_{}".format(feature_name))
    model.save(model_path)

def do_rnn(trainX, testX, trainY, testY):
    global max_document_length
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.1,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="spm-run",n_epoch=5)


def do_mlp(x_train, x_test, y_train, y_test, feature_name):
    # MLP
    model_path = "./Model/MLP/mlp_{}".format(feature_name)
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print clf
    clf.fit(x_train, y_train)
    joblib.dump(clf, model_path)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

def feature_select(X, y):
    return SelectKBest(chi2, k=500).fit_transform(X, y)

if __name__ == "__main__":
    #show_diffrent_max_features()

    #MLP
    #x, y = preprocess.get_features_by_tfidf()
    #feature_name = "tfidf"
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    #do_mlp(x_train, x_test, y_train, y_test, feature_name)

    #CNN
    x, y = preprocess.get_features_by_vt()
    feature_name = "vocabulary table"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    do_cnn(x_train, x_test, y_train, y_test, feature_name)

    #RNN
    #x, y = preprocess.get_features_by_tfidf()
    #x = feature_select(x, y)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
    #do_rnn(x_train, x_test, y_train, y_test)
