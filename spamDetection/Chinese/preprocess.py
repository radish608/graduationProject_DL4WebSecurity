#-*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import jieba

max_features=5000
max_document_length=100

def segment(line):
    '''
    分词
    '''
    return list(jieba.cut(line))

def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x

def load_all_files():
    ham=[]
    spam=[]
    ham = load_files_from_dir('./ham_all.txt')
    spam = load_files_from_dir('./spam_all.txt')
    return ham,spam

def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)

    vectoring = TfidfVectorizer(input='content', tokenizer=segment, analyzer='word')

    x = vectoring.fit_transform(x)





    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y
