# -*- coding:utf-8 -*-
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

max_features=5000

def load_one_file(filename):
    x = ""
    with open(filename) as f:
        for line in f:
            line = line.strip('\n').strip('\r')
            x += line
    return x

def loda_files_from_dir(rootdir):
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham = []
    spam = []
    for i in range(1,7):
        path = "../../Datasets/dataset_mail_English/enron%d/ham" % i
        print "Load %s" % path
        ham +=  loda_files_from_dir(path)
        path = "../../Datasets/dataset_mail_English/enron%d/spam" % i
        print "Load %s" % path
        spam +=  loda_files_from_dir(path)
    return ham, spam

def get_features_by_wordbag():
    ham, spam=load_all_files()
    data = ham + spam
    data2 = list(set(data)) #去重
    #print x
    y = [0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(decode_error='ignore', #"strict", "ignore , "replace"
                                 strip_accents='ascii', #预处理步骤中移除重音的方式
                                 max_features=max_features, #词袋特征个数的最大值
                                 stop_words='english',
                                 max_df=1,
                                 min_df=1)
    print vectorizer
    x1 = vectorizer.fit_transform(data2)
    x1 = x1.toarray()

    #for i in range(len(data)):
        #x2 = vectorizer.transform(x1[i])
    return x1, y

def get_features_by_wordbag_tfjfd():
    transformer = TfidfTransformer(smooth_idf=False)
    x, y = get_features_by_wordbag()
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return x,y

def mlp_wordbag(x_train, x_test, y_train, y_test):
    clf = MLPClassifier(solver="lbfgs",
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    x,y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    mlp_wordbag(x_train, x_test, y_train, y_test)
