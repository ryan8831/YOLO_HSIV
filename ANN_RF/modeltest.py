import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import sys
import os
from tensorflow.keras.datasets import imdb
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tf_slim as slim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from path_def import rainfall_measurement_path_def
import numpy as np
from numpy import matlib
from joblib import load
import pandas as pd
RESULT_ID, READ_PATH, MODEL_READ_PATH, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, WRITE_PATH_MODIFY, KEY_CHAR, SAVE_ARR_IMG, SHADOW_FIX, QC_OPERATION,FORESTER_MODEL_PATH = rainfall_measurement_path_def()
INPUT_PATH = 'F:/Raindrop_folder/Rainfall_project_2023/annlabel/'
test_path = INPUT_PATH +'ANNlabeldataSorted.xlsx'
matplotlib.rc('font', family='Microsoft JhengHei')
# neurons in layers
NUM_FEATURES = 7
L1, L4 = 32, 32
L2, L3 = 64, 64
L5 = 1
L2_BETA = 0.1
SIGMOID_THRESHOLD = 0.6
rf_classifier = load(FORESTER_MODEL_PATH)
WEIGHTS = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,],      #1為使用此資料 0為不使用資料 新版11個資料全部使用就填11個1 
                   dtype = np.bool_).reshape(1, -1)
def batchnorm(Ylogits, is_test, iteration):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean) #True 執行exp_moving_avg.average(mean)  false:mean
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, None, None, bnepsilon)
    return Ybn, update_moving_averages
def make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES, epsilon = 1e-12):
    output = np.array([], dtype = np.float64).reshape(0, (NUM_FEATURES-1))
    former_len = former_data.shape[0]
    latter_len = latter_data.shape[0]
    output = np.vstack((output, np.hstack((former_data, latter_data))))
    velocity = (8.16e-5)*np.sqrt((output[:, 0] - output[:, 3])**2 + (output[:, 1] - output[:, 4])**2)*500    # @@? 8.13e-5 ??
    output = np.hstack((output, velocity.reshape(output.shape[0], 1)))
    output_ = output.copy()  #有_就是原始圖
    output = (output - normalize_para[1][WEIGHTS]) / np.sqrt(normalize_para[0][WEIGHTS] + epsilon)           # @@?
    output /= normalize_para[2][WEIGHTS.squeeze()] # 從數組的形狀中刪除單維度條目，即把shape中為1的維度去掉      # @@?
    # print(output)
    return output, output_, latter_len

# ANN Model=====================================================================
X = tf.placeholder(dtype = tf.float32, shape = [None, NUM_FEATURES], name = 'X')
#==>k_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_keep_prob')
k_prob = tf.placeholder(dtype = tf.float32, name = 'dropout_rate')
# train/test selector for batch normalization
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)
# set L2 Regularizer
regularizer = slim.l2_regularizer(scale = L2_BETA)

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
sess_y = tf.nn.sigmoid(y_logits)
normalize_para = np.load(MODEL_READ_PATH + 'NORMALIZE_PARAMETERS.npy',allow_pickle=True)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
saver = tf.train.Saver(tf.global_variables(), reshape=True)
# saver = tf.train.Saver()
saver.restore(sess, MODEL_READ_PATH + "match_raindrop_10000.ckpt")
data = pd.read_excel(test_path)
test_ann=0
plus_tree=1
choice_class=11
if(test_ann):
    ann_data=data
    former_columns = ['x1', 'y1', 'area1']
    latter_columns = ['x2', 'y2', 'area2']
    former_data = ann_data[former_columns].values
    latter_data = ann_data[latter_columns].values
    y_true=data['Match'].values
    input_data, input_data_, latter_len = make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES)
    actived_y = sess.run(sess_y, feed_dict = {X: input_data, k_prob: 0, tst: True})
    actived_y=(actived_y >= SIGMOID_THRESHOLD).astype(int)
elif(plus_tree):
    tree_data = data[(data['class'] >=choice_class)]
    ann_data=data[(data['class'] < choice_class)]
    former_columns = ['x1', 'y1', 'area1']
    latter_columns = ['x2', 'y2', 'area2']
    former_data = ann_data[former_columns].values
    latter_data = ann_data[latter_columns].values
    y_true=data['Match'].values 
    input_data, input_data_, latter_len = make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES)
    features = pd.DataFrame(tree_data, columns=['x1', 'y1', 'area1', 'a1', 'b1', 'x2', 'y2', 'area2', 'a2', 'b2', 'velocity'])
    features['Area_Difference'] = np.abs(features['area1'] - features['area2'])
    features['Euclidean_Distance'] = np.sqrt((features['x1'] - features['x2'])**2 + (features['y1'] - features['y2'])**2)
    selected_features = features[['Area_Difference','Euclidean_Distance','a1','b1','a2','b2']]
    #selected_features = features[['Area_Difference','Euclidean_Distance']]
    tree_actived_y = rf_classifier.predict(selected_features)
    ann_actived_y = sess.run(sess_y, feed_dict = {X: input_data, k_prob: 0, tst: True})
    ann_actived_y=(ann_actived_y >= SIGMOID_THRESHOLD)
    print(ann_actived_y)
    tree_actived_y = tree_actived_y[:, np.newaxis]
    actived_y = np.vstack((ann_actived_y, tree_actived_y))
else:
    tree_data = data
    y_true=data['Match'].values 
    features = pd.DataFrame(tree_data, columns=['x1', 'y1', 'area1', 'a1', 'b1', 'x2', 'y2', 'area2', 'a2', 'b2', 'velocity'])
    features['Area_Difference'] = np.abs(features['area1'] - features['area2'])
    features['Euclidean_Distance'] = np.sqrt((features['x1'] - features['x2'])**2 + (features['y1'] - features['y2'])**2)
    selected_features = features[['Area_Difference','Euclidean_Distance','a1','b1','a2','b2']]
    tree_actived_y = rf_classifier.predict(selected_features)
    actived_y=tree_actived_y
conf_matrix = confusion_matrix(y_true, actived_y, labels=[1,0])

plt.figure(figsize=(10, 7))
xticklabels = ['1', '0']  
yticklabels = ['1', '0']  

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            annot_kws={"size": 24}, 
            xticklabels=xticklabels, 
            yticklabels=yticklabels)
plt.xlabel('預測標籤', fontsize=30)
plt.ylabel('真實標籤', fontsize=30)
plt.title('混淆矩陣', fontsize=30)
plt.show()
data['modelpredictions'] =actived_y
output_path = INPUT_PATH+'ANNlabeldataPREDICT.xlsx'
data.to_excel(output_path, index=False)