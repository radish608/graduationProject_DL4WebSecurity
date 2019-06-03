# -*- coding: utf-8 -*-

import preprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

class TrainModel():
    def __init__(self, model_path):
        self.characters = preprocess.characters
        self.captcha_char_count = preprocess.captcha_char_count
        self.model_path = model_path
        self.standard_height = preprocess.standard_height
        self.standard_width =preprocess.standard_width

        self.X = tf.placeholder(tf.float32, [None, self.standard_height*self.standard_width])
        self.Y = tf.placeholder(tf.float32, [None, self.captcha_char_count*len(self.characters)])
        self.keep_prob = tf.placeholder(tf.float32)
        self.w_alpha = 0.01
        self.b_alpha = 0.1
    def cnn(self):
        x = tf.reshape(self.X, shape=[-1, self.standard_height, self.standard_width, 1])

        #卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        #卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        #卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'),bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        #全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        #全连接层2
        wout = tf.get_variable(name='wd2', shape=[1024, self.captcha_char_count * len(self.characters)], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.captcha_char_count * len(self.characters)]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict

    def do_cnn(self):
        y_predict = self.cnn()
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        predict = tf.reshape(y_predict, [-1, self.captcha_char_count, len(self.characters)])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.captcha_char_count, len(self.characters)]), 2)

        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #字符
        accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1)) #整个图片

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 恢复模型
            if os.path.exists("./model/cnn"):
                try:
                    saver.restore(sess, self.model_path)
                    print "=====================================ok!"
                # 判断捕获model文件夹中没有模型文件的错误
                except ValueError:
                    print("model is empty")
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                print "----------ok!"

            #x, y = preprocess.get_feature()
            #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
            epoch = 1
            step = 1
            flag = 0
            for j in range(100):
                for i in range(40):
                    x, y = preprocess.getFeatureByBatch(i, size=150)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
                    _, cost_ = sess.run([optimizer, cost], feed_dict={self.X: x_train, self.Y: y_train, self.keep_prob: 0.75})
                    if step % 10 == 0:
                        acc_char = sess.run(accuracy_char_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
                        acc_image = sess.run(accuracy_image_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
                        print("Epoch: {} Step: {} >>> Characters accuracy: {} Image accuracy: {} >>> loss {}".format(epoch, step, acc_char, acc_image, cost_))

                        if acc_image > 0.99:
                            saver.save(sess, self.model_path)
                            x, y = preprocess.get_feature()
                            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
                            acc_char = sess.run(accuracy_char_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
                            acc_image = sess.run(accuracy_image_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
                            if acc_image > 0.95:
                                print("*RESULT: Epoch: {} Step: {} >>> Characters accuracy: {} Image accuracy: {} >>> loss {}".format(epoch, step, acc_char, acc_image, cost_))
                                flag = 1
                                break
                    if i % 500 ==0 :
                        saver.save(sess, self.model_path)
                    step += 1
                if flag == 1:
                    break
                else:
                    epoch += 1
            saver.save(sess, self.model_path)
            sess.close()

    def recognize_captcha(self):
        label, captcha_array = preprocess.img2text(preprocess.dir, random.choice(preprocess.file_list))
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "origin:" + label, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(captcha_array)
        # 预测图片
        image = preprocess.make_gray(captcha_array)
        image = image.flatten() / 255
        y_predict = self.cnn()
        print type(y_predict)
        saver = tf.train.Saver()
        predict = tf.argmax(tf.reshape(y_predict, [-1, self.captcha_char_count, len(self.characters)]), 2)
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            print "======ok!"
            text_list = sess.run(predict, feed_dict={self.X: [image], self.keep_prob: 1.})
            predict_text = text_list[0].tolist()

        print("正确: {}  预测: {}".format(label, predict_text))
        # 显示图片和预测结果
        p_text = ""
        for p in predict_text:
            p_text += str(self.characters[p])
        print(p_text)
        plt.text(20, 1, 'predict:{}'.format(p_text))
        plt.show()

    def show_result(self):
        y_predict = self.cnn()
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        predict = tf.reshape(y_predict, [-1, self.captcha_char_count, len(self.characters)])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.captcha_char_count, len(self.characters)]), 2)

        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #字符
        accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1)) #整个图片

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 恢复模型
            if os.path.exists("./model/cnn"):
                try:
                    saver.restore(sess, self.model_path)
                    print "=====================================ok!"
                # 判断捕获model文件夹中没有模型文件的错误
                except ValueError:
                    print("model is empty")
            else:
                init = tf.global_variables_initializer()
                #sess.run(init)
                print "!!!!!!!!!!!!!!!ERROR!!!!"
                sess.close()
                return

            x, y = preprocess.get_feature()
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
            acc_char = sess.run(accuracy_char_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
            acc_image = sess.run(accuracy_image_count, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
            print("***Characters accuracy: {} Image accuracy: {}".format(acc_char, acc_image))
            sess.close()

if __name__ == '__main__':
    tm = TrainModel("./model/cnn/captcha_model")
    #tm.do_cnn()
    #tm.recognize_captcha()
    tm.show_result()
