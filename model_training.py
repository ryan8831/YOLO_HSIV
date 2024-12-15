#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

# == tf.reset_default_graph()
tf.reset_default_graph()

#INPUT_PATH = 'PATH_TO_LABELED_DATA'
INPUT_PATH = 'H:/PC_ubuntu_space/Raindrop_folder/Rainfall_project_2023/'
#OUTPUT_PATH = 'PATH_TO_TRAINING_RESULT'
OUTPUT_PATH = 'H:/PC_ubuntu_space/Raindrop_folder/Rainfall_project_2023/model/'

# Features: X1, Y1, Area1, a1, b1, canting_angle_1,
#           X2, Y2, Area2, a2, b2, canting_angle_2,
#           Velocity
# NUM_FEATURES: totoal features we consider, but it isn't used in here
NUM_FEATURES = 13

# WEIGHTS: (1) control every parameters in convenient
#          (2) the result below derives from many experiments
# WEIGHTS = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#                    dtype = np.bool).reshape(1, -1)
# a1短軸 b1長軸
WEIGHTS = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,],      #1為使用此資料 0為不使用資料 新版11個資料全部使用就填11個1 
                   dtype = np.bool).reshape(1, -1)

EPOCHES = int(5e4) # int(5e4)
LEARNING_RATE = 1e-3
LR_DECAY_RATIO = 1 - 1e-3
MIN_LEARNING_RATE = 1e-7
BATCH_SIZE = 32
DROP_OUT_PROB = 0.9   #dropout
IS_TEST = False
L2_BETA = 0.1
SIGMOID_THRESHOLD = 0.6

# neurons in layers
L1, L4 = 32, 32
L2, L3 = 64, 64
L5 = 1

# Needed Function===============================================================

def normalize_data(t_p_data_path, t_n_data_path, epsilon = 1e-12):
    t_p_data = pd.read_excel(t_p_data_path,engine='openpyxl')
    t_n_data = pd.read_excel(t_n_data_path,engine='openpyxl')
    data = pd.concat([t_p_data, t_n_data])
    np_data = data.values[:, 0 : -1]

    data_var = np.reshape(np.var(np_data, axis = 0), [1, -1])
    data_mean = np.reshape(np.mean(np_data, axis = 0), [1, -1])
    np_data = (np_data - data_mean) / np.sqrt(data_var + epsilon)
    data_max = np.max(np_data, axis = 0)
    np_data /= data_max
    # print(data_var.shape)
    # print(data_mean.shape)
    # print(data_max.shape)
    normalize_para = (data_var, data_mean, data_max)
    # print(normalize_para)
    return normalize_para, np.concatenate((np_data, np.reshape(data.values[:, -1], [-1, 1])), axis = 1)

import numpy.matlib
def feature_weight(data):
    weights = np.matlib.repmat(WEIGHTS, data.shape[0], 1)
    return data[weights].reshape(-1, np.sum(WEIGHTS.astype(np.int)))

def split_data(data, training_ratio = 0.7):
    dev_ratio = (1 - training_ratio) / 2    # test_ratio = dev_ratio
    np.random.shuffle(data)
    length = len(data)
    #[.......flag1...flag2...end]
    flag1 = int(np.ceil(training_ratio * length))
    flag2 = flag1 + int(np.ceil(dev_ratio * length))

    indices = list(range(0, flag1))
    training_data = data[indices]

    indices = list(range(flag1, flag2))
    dev_data = data[indices]

    indices = list(range(flag2, length))
    test_data = data[indices]
    return training_data, dev_data, test_data

def select_batches(data, batch_size):
    np.random.shuffle(data)
    num_full_batches = int(np.floor(len(data) / batch_size))
    batches = []
    for i in range(num_full_batches):
        batches.append(data[batch_size*i : batch_size*(i + 1), :])
    batches.append(data[batch_size*(i + 1) : -1, :])
    return batches

def calculate_accuracy(data_X, data_y, sess):
    actived_y = sess.run(y, feed_dict = {X: data_X, k_prob: 1, tst: False})
    binary_y = actived_y >= SIGMOID_THRESHOLD
    data_y = data_y == 1
    result = np.logical_not(np.logical_xor(binary_y, data_y))
    return np.mean(result.astype(int))

def batchnorm(Ylogits, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)    # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, None, None, bnepsilon)
    return Ybn, update_moving_averages

# ANN Model=====================================================================

X = tf.placeholder(dtype = tf.float32, shape = [None, np.sum(WEIGHTS.astype(np.int))],
                   name = 'X')
y_label = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = 'y_label')
lr = tf.placeholder(dtype = tf.float32, name = 'learning_rate')
k_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_keep_prob')
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)
# set L2 Regularizer
regularizer = l2_regularizer(scale = L2_BETA)

W1 = tf.get_variable(dtype = tf.float32, shape = [np.sum(WEIGHTS.astype(np.int)), L1],
                     initializer = xavier_initializer(),
                     regularizer = regularizer, name = 'W1')
B1 = tf.get_variable(dtype = tf.float32, shape = [1, L1],
                     initializer = tf.constant_initializer(0.1),
                     regularizer = regularizer, name = 'B1')
W2 = tf.get_variable(dtype = tf.float32, shape = [L1, L2],
                     initializer = xavier_initializer(),
                     regularizer = regularizer, name = 'W2')
B2 = tf.get_variable(dtype = tf.float32, shape = [1, L2],
                     initializer = tf.constant_initializer(0.1),
                     regularizer = regularizer, name = 'B2')
W3 = tf.get_variable(dtype = tf.float32, shape = [L2, L3],
                     initializer = xavier_initializer(),
                     regularizer = regularizer, name = 'W3')
B3 = tf.get_variable(dtype = tf.float32, shape = [1, L3],
                     initializer = tf.constant_initializer(0.1),
                     regularizer = regularizer, name = 'B3')
W4 = tf.get_variable(dtype = tf.float32, shape = [L3, L4],
                     initializer = xavier_initializer(),
                     regularizer = regularizer, name = 'W4')
B4 = tf.get_variable(dtype = tf.float32, shape = [1, L4],
                     initializer = tf.constant_initializer(0.1),
                     regularizer = regularizer, name = 'B4')
W5 = tf.get_variable(dtype = tf.float32, shape = [L4, L5],
                     initializer = xavier_initializer(),
                     regularizer = regularizer, name = 'W5')
B5 = tf.get_variable(dtype = tf.float32, shape = [1, L5],
                     initializer = tf.constant_initializer(0.1),
                     regularizer = regularizer, name = 'B5')

Z1 = tf.matmul(X, W1) + B1
Z1_, update_ema1 = batchnorm(Z1, tst, iter)
A1 = tf.nn.relu(Z1_)
A1 = tf.nn.dropout(A1, keep_prob = k_prob)

Z2 = tf.matmul(A1, W2) + B2
Z2_, update_ema2 = batchnorm(Z2, tst, iter)
A2 = tf.nn.relu(Z2_)
A2 = tf.nn.dropout(A2, keep_prob = k_prob)

Z3 = tf.matmul(A2, W3) + B3
Z3_, update_ema3 = batchnorm(Z3, tst, iter)
A3 = tf.nn.relu(Z3_)
A3 = tf.nn.dropout(A3, keep_prob = k_prob)

Z4 = tf.matmul(A3, W4) + B4
Z4_, update_ema4 = batchnorm(Z4, tst, iter)
A4 = tf.nn.relu(Z4_)
A4 = tf.nn.dropout(A4, keep_prob = k_prob)

y_logits = tf.matmul(A4, W5) + B5
y = tf.nn.sigmoid(y_logits)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_label, logits = y_logits)
regularization_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.reduce_mean(cross_entropy) + regularization_term
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# Main Code=====================================================================
# true_negative_data_path = INPUT_PATH + 'TRUE_NEGATIVE_DATA.xlsx'
# true_positive_data_path = INPUT_PATH + 'TRUE_POSITIVE_DATA.xlsx'

true_negative_data_path = INPUT_PATH + 'true_negative_data_with_vel.xlsx'
true_positive_data_path = INPUT_PATH + 'true_positive_data_with_vel.xlsx'

normalize_para, data = normalize_data(true_positive_data_path, true_negative_data_path, epsilon = 1e-12)
training_data, dev_data, test_data = split_data(data)

# save data
# np.save(OUTPUT_PATH + 'training_data', training_data)
# np.save(OUTPUT_PATH + 'dev_data', dev_data)
# np.save(OUTPUT_PATH + 'test_data', test_data)

training_X = training_data[:, 0 : -1]
training_X = feature_weight(training_X)
training_y = np.reshape(training_data[:, -1], [-1, 1])
training_data = np.hstack((training_X, training_y))
dev_X = dev_data[:, 0 : -1]
dev_X = feature_weight(dev_X)
dev_y = np.reshape(dev_data[:, -1], [-1, 1])
test_X = test_data[:, 0 : -1]
test_X = feature_weight(test_X)
test_y = np.reshape(test_data[:, -1], [-1, 1])

Accuracy, loss_ = [], []
lr_ = LEARNING_RATE
i = 0

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
saver = tf.train.Saver(max_to_keep = 10001)

for epoch in range(1, (EPOCHES + 1)):
    batches = select_batches(training_data, batch_size = BATCH_SIZE)
    lr_ *= LR_DECAY_RATIO if lr_ >= MIN_LEARNING_RATE else MIN_LEARNING_RATE
    i += 1
    for batch in batches:
        batch_X = batch[:, 0 : -1]
        batch_y = np.reshape(batch[:, -1], [-1, 1])
        _ = sess.run(train_op, feed_dict = {X: batch_X, y_label: batch_y, lr: lr_, k_prob: DROP_OUT_PROB, tst: IS_TEST})
        sess.run(update_ema, {X: batch_X, y_label: batch_y, k_prob: DROP_OUT_PROB, tst: IS_TEST, iter: i})

    training_loss = sess.run(loss, feed_dict = {X: training_X, y_label: training_y, k_prob: DROP_OUT_PROB, tst: IS_TEST})
    training_loss = np.mean(training_loss)
    dev_loss = sess.run(loss, feed_dict = {X: dev_X, y_label: dev_y, k_prob: DROP_OUT_PROB, tst: IS_TEST})
    dev_loss = np.mean(dev_loss)
    # record training and dev loss
    loss_.append((training_loss, dev_loss))

    if epoch % 10 == 0 or (epoch + 1) == EPOCHES:
        dev_accuracy = calculate_accuracy(dev_X, dev_y, sess)
        training_accuracy = calculate_accuracy(training_X, training_y, sess)
        Accuracy.append((training_accuracy, dev_accuracy))
        print('Epoch : {:d}/{:d}, Loss : {:.6f}, Traing Accuracy : {:.6f}, Dev Accuracy : {:.6f}'.format(epoch, EPOCHES, training_loss, training_accuracy, dev_accuracy))
        if epoch % 1000 == 0:
            save_path = saver.save(sess, (OUTPUT_PATH + 'match_raindrop_{}.ckpt'.format(epoch)))
            print('Model saved in : {}'.format(save_path))
    else:
        print('Epoch : {:d}/{:d}, Loss : {:.6f}'.format(epoch, EPOCHES, training_loss))
print(normalize_para)
np.save(OUTPUT_PATH + 'NORMALIZE_PARAMETERS', normalize_para)
np.save(OUTPUT_PATH + 'ACCURACY', Accuracy)
np.save(OUTPUT_PATH + 'LOSS', loss_)

# label_y, binary_y, test_accuracy = calculate_accuracy(test_X, test_y, sess)
