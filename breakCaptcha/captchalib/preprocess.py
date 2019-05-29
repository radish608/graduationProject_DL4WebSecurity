#-*- coding:utf-8 -*-
from PIL import Image
import random
import os
import shutil
import sys
import numpy as np
from sklearn.model_selection import train_test_split

captcha_char_count = 4
characters = []
for i in range(11):
    characters.append(str(i))
for i in range(ord('A'), ord('Z')+1):
    characters.append(chr(i))
for i in range(ord('a'), ord('z')+1):
    characters.append(chr(i))

standard_width = 100
standard_height = 60

dir = "./dataset"
file_list = os.listdir(dir)

class TrainError(Exception):
    pass

def check_dataset(dir, standard_width, standard_height, file_suffix):
    print "数据集校验"
    standard_size = (standard_width, standard_height)
    file_count = len(file_list)
    print "数据集共有{}张".format(file_count)

    dataset_error = []

    for i, file_name in enumerate(file_list):
        if not file_name.endswith(file_suffix):
            dataset_error.append((i, file_name, "后缀错误".decode("utf-8")))
            continue
        if len(file_name.split("_")) != 2:
            dataset_error.append((i, file_name, "文件名错误".decode("utf-8")))
            continue

        try:
            file_path = os.path.join(dir, file_name)
            file = Image.open(file_path)
        except OSError:
            dataset_error.append((i, file_name, "文件无法打开".decode("utf-8")))
            continue

        if file.size != standard_size:
            dataset_error.append((i, file_name, "尺寸错误".decode("utf-8")))
            continue

    if dataset_error:
        for i in dataset_error:
            print i;
    else:
        print "pass"
        return dataset_error

def img2text(dir, file_name):
    label = file_name.split("_")[0]
    file_path = os.path.join(dir, file_name)
    file = Image.open(file_path)
    img_array = np.array(file)
    return label, img_array

def make_gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def text2vec(text):
    text_len = len(text)
    if text_len > captcha_char_count:
        raise ValueError('验证码最长{}个字符'.format(captcha_char_count))
    vector = np.zeros(captcha_char_count * len(characters))
    for i, ch in enumerate(text):
        index = i * len(characters) + characters.index(ch)
        vector[index] = 1
    return vector

def getFeatureByBatch(n, size=128):
    batch_x = np.zeros([size, standard_width*standard_height])
    batch_y = np.zeros([size, captcha_char_count * len(characters)])

    max_batch = int(len(file_list) / size)

    if max_batch - 1 < 0:
        raise TrainError("训练集图片数量需要大于每批次训练的图片数量")

    if n > max_batch - 1:
        n = n % max_batch
    s = n * size
    e =  (n + 1) * size
    this_batch = file_list[s:e]

    for i, img_name in enumerate(this_batch):
        label, img_array = img2text(dir, img_name)
        img_array = make_gray(img_array)
        batch_x[i, :] = img_array.flatten() / 255
        batch_y[i, :] = text2vec(label)
    return batch_x, batch_y

def get_feature():
    x = np.zeros([len(file_list), standard_width*standard_height])
    y = np.zeros([len(file_list), captcha_char_count * len(characters)])

    for i, img_name in enumerate(file_list):
        label, img_array = img2text(dir, img_name)
        img_array = make_gray(img_array)
        x[i, :] = img_array.flatten() / 255
        y[i, :] = text2vec(label)
    return x, y


#check_dataset(dir, 160, 60, "png")
