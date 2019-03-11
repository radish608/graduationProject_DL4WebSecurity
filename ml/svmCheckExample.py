#!usr/bin/python
#-*-coding:utf8-*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

#获取样本数据

iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)

#随机分割样本为训练样本和测试样本
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#svm训练
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

#判断预测结果与测试样本标记的结果，得到准确率
print(clf.score(x_test, y_test))

#五折交叉验证
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)
print(scores)
