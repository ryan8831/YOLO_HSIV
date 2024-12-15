# @Author: jibowei
# @Date:   2019-04-25T16:09:16+08:00
# @Filename: rainfall_measurement.py
# @Last modified by:   jibowei
# @Last modified time: 2019-04-25T16:09:16+08:00
# @License: MIT


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import csv
import time
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
import tf_slim as slim
from skimage import morphology
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import bounding_para, make_input_data, draw_arrow, Rainfall_Meter, refine_file_list, plot_confusion_matrix
from utils import check_out_path_exist, csv_to_xlsx_pd, fnc_show_img
from path_def import rainfall_measurement_path_def

#tf.reset_default_graph()
tf.reset_default_graph # ==>

### =========
from tensorflow.compat.v1 import keras
from keras.datasets import imdb
import tensorflow_datasets as tfds
filename="0321紀錄時間.txt"
def record_time(start):
    if start:
        t="開始"
    else:
        t="結束"
    with open(filename, 'a') as file:
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        file.write("程式"+t+f"時間: {end_time}\n")

record_time(1)
dataset, information = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
# print(dataset)
train_dataset, test_dataset = dataset['train'], dataset['test']

#np_load_old = np.load
# modify the default parameters of np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true

np.load.__defaults__=(None, True, True, 'ASCII')

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
#np.load = np_load_old
np.load.__defaults__=(None, False, True, 'ASCII')




### ========
# PATH DEFINE
REAULT_ID, READ_PATH, MODEL_READ_PATH, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, WRITE_PATH_MODIFY, KEY_CHAR = rainfall_measurement_path_def()

### ========

check_out_path_exist(WRITE_PATH, REAULT_ID)

### ========
excel1='雨滴每秒結果'+ REAULT_ID
excel2='雨滴小時結果'+ REAULT_ID
excel3='雨滴分鐘結果'+ REAULT_ID


with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel1),'w',newline='',encoding="utf-8") as f1:
            writer=csv.writer(f1)
            writer.writerow(["檔案名稱","X","Y","area","短軸","長軸","X差值","Y差值","速度","直徑"])
with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel2),'w',newline='',encoding="utf-8") as f2:
            writer=csv.writer(f2)
            writer.writerow(["第幾個小時","總雨量","總數量","累計總雨量","累計總數量"])
with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel3),'w',newline='',encoding="utf-8") as f3:
            writer=csv.writer(f3)
            writer.writerow(["第幾分鐘","總雨量","總數量","累計總雨量","累計總數量"])

# neurons in layers
NUM_FEATURES = 7
L1, L4 = 32, 32
L2, L3 = 64, 64
L5 = 1

L2_BETA = 0.01
SIGMOID_THRESHOLD = 0.7
CLOSE_KERNEL = 7
ACCEPT_DIS = 50     #雨滴下一偵實際位置與預測落點的“可接受距離”

def batchnorm(Ylogits, is_test, iteration):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, None, None, bnepsilon)
    return Ybn, update_moving_averages

def match_raindrop(input_data, latter_len, sess, SIGMOID_THRESHOLD):
    if input_data.shape[0] != 0:

        actived_y = sess.run(y, feed_dict = {X: input_data, k_prob: 0, tst: True})  #Arrow_R_20170616_021421_492_493
        actived_y = np.reshape(actived_y, [-1, latter_len])
        #actived_y_ = actived_y
        max_value_row = np.max(actived_y, axis = 1).reshape((-1, 1))
        max_value_col = np.max(actived_y, axis = 0).reshape((1, -1))
        intersection_max_mask = (actived_y == max_value_row) * (actived_y == max_value_col)
        intersection_max_mask = (actived_y >= SIGMOID_THRESHOLD) * intersection_max_mask
        # print('intersection_max_mask={}'.format(intersection_max_mask))
        add = np.invert(np.sum(intersection_max_mask, axis = 1).astype(bool))
        add_mask = np.matlib.repmat(add.reshape(-1, 1), 1, latter_len)
        # print('add_mask ={}'.format(add_mask ))
        choose_mask = np.invert(np.sum(intersection_max_mask, axis = 0).astype(bool)).reshape(1, -1)
        add_data = actived_y * add_mask
        add_mask = (add_data == np.max(add_data, axis = 0)) * add_data
        add_mask = (choose_mask * add_mask) >= SIGMOID_THRESHOLD
        # print('add_mask ={}'.format(np.argmax(add_mask*actived_y , axis = 1)))     #要改
        tmp_add_mask=np.zeros_like(intersection_max_mask)
        tmp=np.argmax(add_mask*actived_y , axis = 1)
        for i in range(0,len(tmp)):
            tmp_add_mask[i][tmp[i]]=(add_mask*actived_y)[i][tmp[i]]
        # print('tmp_add_mask ={}'.format(tmp_add_mask)) 
        actived_y = np.multiply(actived_y, (intersection_max_mask + tmp_add_mask))
        actived_y = actived_y.reshape((-1, 1))
        
        return (actived_y >= SIGMOID_THRESHOLD)
    else:
        return np.zeros((1, 1), dtype = bool)

# ANN Model=====================================================================
X = tf.placeholder(dtype = tf.float32, shape = [None, NUM_FEATURES], name = 'X')
#==>k_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_keep_prob')
k_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_rate')
# train/test selector for batch normalization
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)
# set L2 Regularizer
#regularizer = l2_regularizer(scale = L2_BETA)
regularizer = slim.l2_regularizer(scale = L2_BETA)

# W1 = tf.get_variable(dtype = tf.float32, shape = [NUM_FEATURES, L1], initializer =
#                      xavier_initializer(), regularizer = regularizer, name = 'W1')
# B1 = tf.get_variable(dtype = tf.float32, shape = [1, L1], initializer =
#                      tf.constant_initializer(0.1), regularizer = regularizer, name = 'B1')
# W2 = tf.get_variable(dtype = tf.float32, shape = [L1, L2], initializer =
#                      xavier_initializer(), regularizer = regularizer, name = 'W2')
# B2 = tf.get_variable(dtype = tf.float32, shape = [1, L2], initializer =
#                      tf.constant_initializer(0.1), regularizer = regularizer, name = 'B2')
# W3 = tf.get_variable(dtype = tf.float32, shape = [L2, L3], initializer =
#                      xavier_initializer(), regularizer = regularizer, name = 'W3')
# B3 = tf.get_variable(dtype = tf.float32, shape = [1, L3], initializer =
#                      tf.constant_initializer(0.1), regularizer = regularizer, name = 'B3')
# W4 = tf.get_variable(dtype = tf.float32, shape = [L3, L4], initializer =
#                      xavier_initializer(), regularizer = regularizer, name = 'W4')
# B4 = tf.get_variable(dtype = tf.float32, shape = [1, L4], initializer =
#                      tf.constant_initializer(0.1), regularizer = regularizer, name = 'B4')
# W5 = tf.get_variable(dtype = tf.float32, shape = [L4, L5], initializer =
#                      xavier_initializer(), regularizer = regularizer, name = 'W5')
# B5 = tf.get_variable(dtype = tf.float32, shape = [1, L5], initializer =
#                      tf.constant_initializer(0.1), regularizer = regularizer, name = 'B5')
W1 = tf.get_variable(dtype = tf.float32, shape = [NUM_FEATURES, L1], initializer =
                     tf.initializers.glorot_uniform(), regularizer = regularizer, name = 'W1')
B1 = tf.get_variable(dtype = tf.float32, shape = [1, L1], initializer =
                     tf.constant_initializer(0.1), regularizer = regularizer, name = 'B1')
W2 = tf.get_variable(dtype = tf.float32, shape = [L1, L2], initializer =
                     tf.initializers.glorot_uniform(), regularizer = regularizer, name = 'W2')
B2 = tf.get_variable(dtype = tf.float32, shape = [1, L2], initializer =
                     tf.constant_initializer(0.1), regularizer = regularizer, name = 'B2')
W3 = tf.get_variable(dtype = tf.float32, shape = [L2, L3], initializer =
                     tf.initializers.glorot_uniform(), regularizer = regularizer, name = 'W3')
B3 = tf.get_variable(dtype = tf.float32, shape = [1, L3], initializer =
                     tf.constant_initializer(0.1), regularizer = regularizer, name = 'B3')
W4 = tf.get_variable(dtype = tf.float32, shape = [L3, L4], initializer =
                     tf.initializers.glorot_uniform(), regularizer = regularizer, name = 'W4')
B4 = tf.get_variable(dtype = tf.float32, shape = [1, L4], initializer =
                     tf.constant_initializer(0.1), regularizer = regularizer, name = 'B4')
W5 = tf.get_variable(dtype = tf.float32, shape = [L4, L5], initializer =
                    tf.initializers.glorot_uniform(), regularizer = regularizer, name = 'W5')
B5 = tf.get_variable(dtype = tf.float32, shape = [1, L5], initializer =
                     tf.constant_initializer(0.1), regularizer = regularizer, name = 'B5')
Z1 = tf.matmul(X, W1) + B1
Z1_, update_ema1 = batchnorm(Z1, tst, iter)
A1 = tf.nn.relu(Z1_)
#==>A1 = tf.nn.dropout(A1, keep_prob = k_prob)
A1 = tf.nn.dropout(A1, rate = k_prob)

Z2 = tf.matmul(A1, W2) + B2
Z2_, update_ema2 = batchnorm(Z2, tst, iter)
A2 = tf.nn.relu(Z2_)
#==>A2 = tf.nn.dropout(A2, keep_prob = k_prob)
A2 = tf.nn.dropout(A2, rate = k_prob)


Z3 = tf.matmul(A2, W3) + B3
Z3_, update_ema3 = batchnorm(Z3, tst, iter)
A3 = tf.nn.relu(Z3_)
#==>A3 = tf.nn.dropout(A3, keep_prob = k_prob)
A3 = tf.nn.dropout(A3, rate = k_prob)

Z4 = tf.matmul(A3, W4) + B4
Z4_, update_ema4 = batchnorm(Z4, tst, iter)
A4 = tf.nn.relu(Z4_)
#==>A4 = tf.nn.dropout(A4, keep_prob = k_prob)
A4 = tf.nn.dropout(A4, rate = k_prob)

y_logits = tf.matmul(A4, W5) + B5
y = tf.nn.sigmoid(y_logits)

# Main Code=====================================================================
#rename_files(READ_PATH)
print(READ_PATH)
file_list = sorted(os.listdir(path = READ_PATH))

file_list.append('0'*25)
normalize_para = np.load(MODEL_READ_PATH + 'NORMALIZE_PARAMETERS.npy',allow_pickle=True)
# print(normalize_para)
closing_kernel = morphology.disk(radius = CLOSE_KERNEL)#圓型https://www.cnblogs.com/denny402/p/5132677.html
plt
rainfall_agent = Rainfall_Meter(accept_dis = ACCEPT_DIS, ana_path = WRITE_PATH_ANALYZE)
# set a blank former image for beginning
former_name = '9'*25 #讓它們可以相減
former_img = np.zeros((640, 480, 3), dtype = np.uint8)
former_data = np.array([], dtype = np.float64).reshape(0, NUM_FEATURES) #reshape x列x行 0,7
former_ellipse_axis = np.zeros((1, 2))
former_ellipse_axis = np.zeros((1, 2))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
# read trained model

saver = tf.train.Saver(tf.global_variables(), reshape=True)
# saver = tf.train.Saver()
saver.restore(sess, MODEL_READ_PATH + "match_raindrop_30000.ckpt")

img_pair_counter, img_counter = 0, 0
fig = plt.figure(num = 1, dpi = 500, frameon = False)



for latter_name in file_list:
    # if the latter image is the next one from former image
    if latter_name[0] == 'R' and int(latter_name[18 : 21]) - int(former_name[18 : 21]) == 1:

        ax = plt.axes()
        ax.set_axis_off()
        compose_img_name = former_name[0 : 21] + '_' + latter_name[18 : 21] + '.png'

        # read latter image and extract wanted data
        latter_img = np.transpose(cv2.imread(READ_PATH + latter_name), axes = (1, 0, 2))   # transpose函數里，軸默認順序是z，x，y，分別對應0軸，1軸和2軸
        latter_data, latter_ellipse_axis, rects, boxes = bounding_para(latter_img, closing_kernel)
        # input的圖像有幾個區域，data就會有幾列
        # print(latter_data)
        # print(normalize_para.shape)
        # synthesize former and latter image
        r_latter_img = latter_img.copy()
        r_latter_img[:, :, 1 : 3] = 0
        compose_img = former_img + r_latter_img
        compose_img = np.transpose(compose_img, axes = (1, 0, 2))
        ax.imshow(compose_img)

        # get input data for ANN model
        input_data, input_data_, latter_len = make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES)      ## trace
        # if the image is not blank, then HAVE_DATA is true
        HAVE_DATA = True if input_data_.shape[0] != 0 else False
        # match raindrop pair using ANN model and return binary results
        binary_y = match_raindrop(input_data, latter_len, sess, SIGMOID_THRESHOLD)
        # calibrate results using preceding tracking path
        matched_pair = rainfall_agent.calibrating_path(input_data_,
                                                       binary_y,
                                                       (latter_data, latter_ellipse_axis),
                                                       compose_img_name,
                                                       NUM_FEATURES
                                                       )
        # print(matched_pair)
        draw_arrow(matched_pair, [], [], fig, ax)

        arrow_img_name = 'Arrow_' + compose_img_name
        fig.savefig( WRITE_PATH_ARRIMG + arrow_img_name, transparent = True, pad_inches = 0)
        fig.clear()

        img_pair_counter += 1
        if img_pair_counter % 100 == 0: print('{:d}, {:s}'.format(img_pair_counter, compose_img_name))
        # LAST_IMG is true when the image is the last one in that second
        LAST_IMG = True if int(file_list[img_counter + 1][18 : 21]) - int(latter_name[18 : 21]) != 1 else False
        # LAST_HR is true when the image is the last one in that hour
        # EX:R_20170616_000006_033.png    20170616日期_00小時0006_分鐘033秒
        LAST_HR = True if int(file_list[img_counter + 1][11 : -12]) - int(latter_name[11 : -12]) == 1 else False
        # print(LAST_IMG)
        # print(HAVE_DATA)
        # print(latter_name[0 : 21])
        # print(LAST_HR)
        if LAST_IMG: rainfall_agent.record_last_img_raindrop_info(latter_name[0 : 21] + '_000.png', HAVE_DATA)
        if LAST_HR: 
            rainfall_agent.record_rainfall_in_hour(compose_img_name)
            # rainfall_agent.count_raindrops={}
            # rainfall_agent.full_memory={}

        # reset parameters
        former_name = latter_name
        former_img = latter_img
        former_data = latter_data
        former_ellipse_axis = latter_ellipse_axis
    elif latter_name[0] == 'R':
        rainfall_agent.reset_memory()
        former_name = latter_name
        former_img = np.transpose(cv2.imread(READ_PATH + former_name), axes = (1, 0, 2))
        former_data, _, _, _ = bounding_para(former_img, closing_kernel)
        former_ellipse_axis = np.zeros((1, 2))
    else:
        # give a fake name
        former_name = '9'*25
    img_counter += 1
    if ((img_counter % 500) == 0 ) :
          print("---------------------\n")
          print("img_counter = {}",img_counter)
          print("---------------------\n")
    csv_to_xlsx_pd(excel1, WRITE_PATH_ANALYZE)
    csv_to_xlsx_pd(excel2, WRITE_PATH_ANALYZE)
    csv_to_xlsx_pd(excel3, WRITE_PATH_ANALYZE)
record_time(0)