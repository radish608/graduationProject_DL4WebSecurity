# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import re
from sklearn.externals import joblib
import pickle

def load_all_files():
    import glob
    x=[]
    y=[]
    #加载攻击样本
    files=glob.glob("../Datasets/dataset_linuxbackdoor/ADFA-LD/Attack_Data_Master/*/*")
    for file in files:
        with open(file) as f:
            lines=f.readlines()
        x.append(" ".join(lines))
        y.append(1)
    print "Load black data %d" % len(x)
    #加载正常样本
    files=glob.glob("../Datasets/dataset_linuxbackdoor/ADFA-LD/Training_Data_Master/*")
    for file in files:
        with open(file) as f:
            lines=f.readlines()
        x.append(" ".join(lines))
        y.append(0)
    print "Load full data %d" % len(x)

    return x,y

def get_feature_wordbag(n):
    save_name = "./Model/Data/wordbag_{}-Gram.data".format(n)
    if os.path.exists(save_name):
        f = open(save_name,'r')
        x, y, transformer = pickle.loads(f.read())
        f.close()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
        return x_train, x_test, y_train, y_test

    max_features=1000
    x,y=load_all_files()
    vectorizer = CountVectorizer(
                                 ngram_range=(n, n),
                                 token_pattern=r'\b\d+\b',
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print vectorizer
    x = vectorizer.fit_transform(x)

    transformer = TfidfTransformer(smooth_idf=False)
    x=transformer.fit_transform(x)

    x = x.toarray()

    data = pickle.dumps((x, y, transformer))

    with open(save_name, 'w') as f:
        f.write(data)
        f.close()


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test
