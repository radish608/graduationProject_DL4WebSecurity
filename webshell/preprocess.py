# -*-coding: utf-8 -*-
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import commands
import tflearn
import pickle

max_features=10000
max_document_length=100
min_opcode_count=2

webshell_dir="../Datasets/dataset_webshell/webshell/PHP/"
whitefile_dir="../Datasets/dataset_webshell/normal/php/"
check_dir="~/Downloads/php-exploit-scripts/"
white_count=0
black_count=0
php_bin="/usr/bin/php"

def load_files_re(dir):
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') or filename.endswith('.txt'):
                fulepath = os.path.join(path, filename)
                print "Load %s" % fulepath
                t = load_file(fulepath)
                files_list.append(t)

    return files_list

def load_files_opcode_re(dir):
    global min_opcode_count
    files_list = []
    g = os.walk(dir)
    for path, d, filelist in g:
        #print d;
        for filename in filelist:
            #print os.path.join(path, filename)
            if filename.endswith('.php') :
                fulepath = os.path.join(path, filename)
                print "Load %s opcode" % fulepath
                t = load_file_opcode(fulepath)
                if len(t) > min_opcode_count:
                    files_list.append(t)
                else:
                    print "Load %s opcode failed" % fulepath
                #print "Add opcode %s" % t

    return files_list


def load_file(file_path):
    t=""
    with open(file_path) as f:
        for line in f:
            line=line.strip('\n')
            t+=line
    return t

def load_file_opcode(file_path):
    global php_bin
    t=""
    cmd=php_bin+" -dvld.active=1 -dvld.execute=0 "+file_path
    #print "exec "+cmd
    status,output=commands.getstatusoutput(cmd)

    t=output
        #print t
    tokens=re.findall(r'\s(\b[A-Z_]+\b)\s',output)
    t=" ".join(tokens)

    print "opcode count %d" % len(t)
    return t

def load_files(path):
    files_list=[]
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path=path+file
                print "Load %s" % file_path
                t=load_file(file_path)
                files_list.append(t)
    return  files_list

#php N-Gram + TF-IDF
def get_feature_by_ngram():
    global white_count
    global black_count
    global max_features
    print "max_features=%d" % max_features
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    y=y1+y2

    CV = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    #x_tfidf = transformer.fit_transform(x)
    #x = x_tfidf.toarray()

    return x,y


#opcode N-Gram
def get_feature_by_opcode_ngram():
    global white_count
    global black_count
    global max_features
    print "max_features=%d" % max_features
    x=[]
    y=[]

    data_file = "./Model/Data/opcode_ngram_tf.data"

    if os.path.exists(data_file):
        f = open(data_file, 'rb')
        x, y = pickle.loads(f.read())
        f.close()

        return x, y

    webshell_files_list = load_files_opcode_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_opcode_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    y=y1+y2

    CV = CountVectorizer(ngram_range=(2, 4), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)
    x=CV.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    #x_tfidf = transformer.fit_transform(x)
    #x = x_tfidf.toarray()

    data = pickle.dumps((x, y))

    with open(data_file, 'w') as f:
        f.write(data)
        f.close()

    f.close()

    return x,y


#opcode词汇表
def get_feature_by_opcode_vt():
    global white_count
    global black_count
    global max_document_length
    x=[]
    y=[]
    data_file = "./Model/Data/opcode_vt.data"

    if os.path.exists(data_file):
        f = open(data_file, 'rb')
        x, y = pickle.loads(f.read())
        f.close()

    else:
        webshell_files_list = load_files_opcode_re(webshell_dir)
        y1=[1]*len(webshell_files_list)
        black_count=len(webshell_files_list)

        wp_files_list =load_files_opcode_re(whitefile_dir)
        y2=[0]*len(wp_files_list)

        white_count=len(wp_files_list)


        x=webshell_files_list+wp_files_list
        #print x
        y=y1+y2

        vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                  min_frequency=0,
                                                  vocabulary=None,
                                                  tokenizer_fn=None)
        x=vp.fit_transform(x, unused_y=None)
        x=np.array(list(x))

        f = open(data_file, 'wb')
        data = pickle.dumps((x, y))
        f.write(data)
        f.close()
    #print x
    #print y
    return x,y


#php词汇表
def  get_feature_by_vt():
    global max_document_length
    global white_count
    global black_count
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)


    x=webshell_files_list+wp_files_list
    y=y1+y2

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    return x,y


#php序列
def get_feature_by_php():
    global white_count
    global black_count
    global max_features
    global webshell_dir
    global whitefile_dir
    print "max_features=%d webshell_dir=%s whitefile_dir=%s" % (max_features,webshell_dir,whitefile_dir)
    x=[]
    y=[]

    webshell_files_list = load_files_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    #print x
    y=y1+y2

    CV = CountVectorizer(ngram_range=(3000, 3000), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)

    x=CV.fit_transform(x).toarray()

    return x,y

#opcode序列
def get_feature_by_opcode():
    global white_count
    global black_count
    global max_features
    global webshell_dir
    global whitefile_dir
    print "max_features=%d webshell_dir=%s whitefile_dir=%s" % (max_features,webshell_dir,whitefile_dir)
    x=[]
    y=[]

    data_file = "./Model/Data/opcodelist.data"

    if os.path.exists(data_file):
        f = open(data_file, 'rb')
        x, y = pickle.loads(f.read())
        f.close()

        return x, y

    webshell_files_list = load_files_opcode_re(webshell_dir)
    y1=[1]*len(webshell_files_list)
    black_count=len(webshell_files_list)

    wp_files_list =load_files_opcode_re(whitefile_dir)
    y2=[0]*len(wp_files_list)

    white_count=len(wp_files_list)

    x=webshell_files_list+wp_files_list
    #print x
    y=y1+y2

    CV = CountVectorizer(ngram_range=(3000, 3000), decode_error="ignore",max_features=max_features,
                                       token_pattern = r'\b\w+\b',min_df=1, max_df=1.0)

    x=CV.fit_transform(x).toarray()

    f = open(data_file, 'wb')
    data = pickle.dumps((x, y))
    f.write(data)
    f.close()

    return x,y
