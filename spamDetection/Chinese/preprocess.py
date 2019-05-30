#-*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import jieba
import tflearn

max_features=5000
max_document_length=100

def segment(line):
    '''
    分词
    '''
    return list(jieba.cut(line))

def get_features_by_tfidf():
    global  max_document_length
    file_list = ['./ham_5000.utf8', './spam_5000.utf8']

    content = []
    file_cnt = []
    for file_name in file_list:
        before_size = len(content)
        content.extend(open(file_name,'r').readlines())
        file_cnt.append(len(content)-before_size)
    """
    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=100,
                                                  min_frequency=0,
                                                  vocabulary=None,
                                                  tokenizer_fn=segment)
    """

    vectoring = TfidfVectorizer(input='content', tokenizer=segment, analyzer='word')
    x = vectoring.fit_transform(content)
    y = np.concatenate((np.repeat([1], file_cnt[0],axis=0),
                        np.repeat([0], file_cnt[1], axis=0)), axis=0)

    x = x.toarray()
    return x, y


def get_features_by_tf():
    global max_document_length
    x=[]
    y=[]
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    return x,y
