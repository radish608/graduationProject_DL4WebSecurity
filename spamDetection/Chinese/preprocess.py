# -*- coding: utf-8 -*-

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
from hanziconv import HanziConv
import pickle

max_features=5000
data_model_path = "./Model/DATA/{}.data"

def segment(line):
    return list(jieba.cut(line))

def segmentA(documents):
    for document in documents:
        # 繁体转简体
        text = HanziConv.toSimplified(document)
        # 英文转小写
        text = text.lower()
        # 分词
        yield list(jieba.cut(text))
    #line = HanziConv.toSimplified(line)
    #line = line.lower()
    #return list(jieba.cut(line))

def get_features_by_tfidf():
    data_path = data_model_path.format("tfidf")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            x, y = pickle.loads(f.read())
            f.close()
        return x, y

    file_list = ['./ham_5000.utf8', './spam_5000.utf8']

    content = []
    file_cnt = []
    for file_name in file_list:
        before_size = len(content)
        content.extend(open(file_name,'r').readlines())
        file_cnt.append(len(content)-before_size)

    vectoring = TfidfVectorizer(input='content', tokenizer=segment, analyzer='word')
    x = vectoring.fit_transform(content)
    y = np.concatenate((np.repeat([1], file_cnt[0],axis=0),
                        np.repeat([0], file_cnt[1], axis=0)), axis=0)

    x = x.toarray()
    #data = pickle.dumps((x, y))
    #with open(data_path, "w") as f:
    #    f.write(data)
    #    f.close()
    return x, y


def get_features_by_vt():
    data_path = data_model_path.format("vt")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            x, y = pickle.loads(f.read())
            f.close()
        return x, y

    max_document_length=500
    file_list = ['./ham_5000.utf8', './spam_5000.utf8']

    content = []
    file_cnt = []
    for file_name in file_list:
        before_size = len(content)
        content.extend(open(file_name,'r').readlines())
        file_cnt.append(len(content)-before_size)
    y = np.concatenate((np.repeat([1], file_cnt[0],axis=0),
                        np.repeat([0], file_cnt[1], axis=0)), axis=0)
    vp=tflearn.data_utils.VocabularyProcessor(max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=segmentA)
    x=vp.fit_transform(content)
    x=np.array(list(x))
    data = pickle.dumps((x, y))
    with open(data_path, "w") as f:
        f.write(data)
        f.close()
    return x, y

#get_features_by_tf()

#print segmentA("天气不错")
