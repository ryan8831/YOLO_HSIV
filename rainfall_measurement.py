# @Author: CCU LAB123 raindrop team
# @Filename: rainfall_measurement.py
# @Last modified by:   CCU LAB123 raindrop team

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import cv2
import time
import matplotlib
import numpy                as np
import pandas               as pd
import tf_slim              as slim
import matplotlib.pyplot    as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
matplotlib.use('Agg')
from joblib import load
from skimage         import morphology
from path_def        import rainfall_measurement_path_def
from km_utils     import *
from ultralytics import YOLO
#tf.reset_default_graph()
tf.reset_default_graph # ==>

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

def match_raindrop(input_data, latter_len, sess, SIGMOID_THRESHOLD,input_data_
                   ,rf_classifier,former_data,former_ellipse_axis,latter_data,latter_ellipse_axis,
                   sess_y,X,k_prob,tst):
    if input_data.shape[0] != 0:
        actived_y = sess.run(sess_y, feed_dict = {X: input_data, k_prob: 0, tst: True})
        fulldata=np.vstack((np.hstack((former_data, former_ellipse_axis)),np.hstack((latter_data, latter_ellipse_axis))))
        axis_mapping = {(row[0], row[1], row[2]): row[3:5] for row in fulldata}
        new_input_data = []
        for row in input_data_:
            x1, y1, area1, x2, y2, area2, velocity = row
            # 從字典中查找對應的長短軸
            axes1 = axis_mapping.get((x1, y1, area1))
            axes2 = axis_mapping.get((x2, y2, area2))
            f_diameter=6.486e-2*((axes1[1]**2 * axes1[0])**(1 / 3))
            l_diameter=6.486e-2*((axes2[1]**2 * axes2[0])**(1 / 3))
            diameter=f_diameter if f_diameter>l_diameter else l_diameter
            new_row = [x1, y1, area1, axes1[1],axes1[0], x2, y2, area2, axes2[1],axes2[0], velocity,diameter]
            new_input_data.append(new_row)
        new_input_data = np.array(new_input_data)   
        mask = (input_data_[:, -1] >=1.25)
        random_forest_data = new_input_data[mask]
        if random_forest_data.size > 0:
            features = pd.DataFrame(random_forest_data, columns=['x1', 'y1', 'area1', 'a1', 'b1', 'x2', 'y2', 'area2', 'a2', 'b2', 'velocity','diameter'])
            features['Area_Difference'] = np.abs(features['area1'] - features['area2'])
            features['Euclidean_Distance'] = np.sqrt((features['x1'] - features['x2'])**2 + (features['y1'] - features['y2'])**2)
            selected_features = features[['Area_Difference','Euclidean_Distance','a1','b1','a2','b2']]
            #selected_features = features[['Area_Difference']]
            y_pred = rf_classifier.predict(selected_features)
            positive_indices = np.where(y_pred == 1)[0]
            ori_indices = np.where(mask)[0][positive_indices]
            actived_y[ori_indices] = 1
        actived_y = np.reshape(actived_y, [-1, latter_len])
        row_ind, col_ind = linear_sum_assignment(-actived_y)
        intersection_max_mask = np.zeros_like(actived_y, dtype=bool)
        intersection_max_mask[row_ind, col_ind] = True
        actived_y = np.multiply(actived_y, (intersection_max_mask))
        actived_y = actived_y.reshape((-1, 1))    
        return (actived_y >= SIGMOID_THRESHOLD)
    else:
        return np.zeros((1, 1), dtype = bool)
def calculate_diameter(elips_ax):
    return (8.13e-2) * ((elips_ax[:, 1]**2 * elips_ax[:, 0])**(1 / 3))
def find_matching_indices(data, subarray):
    return [np.where(np.all(np.isclose(data, row), axis=1))[0][0] if np.any(np.all(np.isclose(data, row), axis=1)) else None for row in subarray]
# ANN Model=====================================================================
def ANN_model():
    # ======== neurons in layers
    L1, L4            = 32, 32
    L2, L3            = 64, 64
    L5                = 1
    L2_BETA           = 0.01
    NUM_FEATURES      = 7
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
    sess_y   = tf.nn.sigmoid(y_logits)
    return sess_y,X,k_prob,tst
def yolo_bounding(ori_img,yolov9det):
    output = np.array([], dtype=np.float64).reshape(0, 3)
    ellipse_axis = np.array([], dtype=np.float64).reshape(0, 2)
    results = yolov9det.predict(ori_img)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    for (x1, y1, x2, y2), conf,cls in zip(boxes, confidences,classes):
        if(cls==1 and conf>0.2):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            elps_ax = np.array([w, h]).reshape(1, -1)
            elps_ax.sort()
            area = np.pi * elps_ax[0][0] * elps_ax[0][1] / 4
            rect_ = np.array([cx, cy, area]).reshape(1, 3)
            output = np.vstack((output, rect_))
            ellipse_axis = np.vstack((ellipse_axis, elps_ax))
    return output, ellipse_axis
def F_rainfall_measurement(USEYOLO):
    ### ========
    # PATH DEFINE
    RESULT_ID, READ_PATH, MODEL_READ_PATH, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, WRITE_PATH_MODIFY, KEY_CHAR, SAVE_ARR_IMG, SHADOW_FIX, QC_OPERATION,FORESTER_MODEL_PATH = rainfall_measurement_path_def()
    check_out_path_exist(WRITE_PATH, RESULT_ID)
    ### ========
    NUM_FEATURES      = 7
    sess_y,X,k_prob,tst=ANN_model()
    excel1='雨滴每秒結果'+ RESULT_ID
    excel2='雨滴小時結果'+ RESULT_ID
    excel3='雨滴分鐘結果'+ RESULT_ID
    CLOSE_KERNEL      = 7
    SIGMOID_THRESHOLD = 0.6
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel1),'w',newline='',encoding="utf-8") as f1:
                writer=csv.writer(f1)
                writer.writerow(["檔案名稱","X","Y","area","短軸","長軸","X差值","Y差值","速度","直徑","角度","角度變化量"])
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel2),'w',newline='',encoding="utf-8") as f2:
                writer=csv.writer(f2)
                writer.writerow(["第幾個小時","總雨量","總數量","累計總雨量","累計總數量"])
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(excel3),'w',newline='',encoding="utf-8") as f3:
                writer=csv.writer(f3)
                writer.writerow(["第幾分鐘","總雨量","總數量","累計總雨量","累計總數量"])

    # Main Code=====================================================================
    start_time = time.time()
    #rename_files(READ_PATH)
    print("\n=========================================================================================================\n")
    print(READ_PATH)
    print("\n=========================================================================================================\n")
    file_list = sorted(os.listdir(path = READ_PATH))

    file_list.append('0'*25)
    normalize_para = np.load(MODEL_READ_PATH + 'NORMALIZE_PARAMETERS.npy',allow_pickle=True)
    # print(normalize_para)
    closing_kernel = morphology.disk(radius = CLOSE_KERNEL)
    plt
    rainfall_agent = Rainfall_Meter(ana_path = WRITE_PATH_ANALYZE)
    # set a blank former image for beginning
    former_name = '9'*25
    former_img = np.zeros((640, 480, 3), dtype = np.uint8)
    former_data = np.array([], dtype = np.float64).reshape(0, NUM_FEATURES)
    former_ellipse_axis = np.zeros((1, 2))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # read trained model

    saver = tf.train.Saver(tf.global_variables(), reshape=True)
    # saver = tf.train.Saver()
    saver.restore(sess, MODEL_READ_PATH + "match_raindrop_10000.ckpt")

    img_pair_counter, img_counter = 0, 0
    fig = plt.figure(num = 1, dpi = 500, frameon = False)

    tracked_memory     ={}
    first_new_raindrop ={}
    output_memory      ={}
    ex_matched=np.empty((0,0))
    rf_classifier = load(FORESTER_MODEL_PATH)
    if USEYOLO==True:
        YOLOv9det=YOLO('D:/github/python_code_source/yolov9model/yolov9s-9281.pt')
    rainclass=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, float('inf')]

    for latter_name in file_list:
        # if the latter image is the next one from former image
        if latter_name[0] == KEY_CHAR and int(latter_name[18 : 21]) - int(former_name[18 : 21]) == 1:

            if(SAVE_ARR_IMG):
                ax = plt.axes()
                ax.set_axis_off()

            compose_img_name = former_name[0 : 21] + '_' + latter_name[18 : 21] + '.png'

            # read latter image and extract wanted data
            latter_img_o=cv2.imread(READ_PATH + latter_name)
            latter_img = np.transpose(latter_img_o, axes = (1, 0, 2))   # transpose函數里，軸默認順序是z，x，y，分別對應0軸，1軸和2軸
            if USEYOLO==True :
                latter_data,latter_ellipse_axis=yolo_bounding(latter_img_o,YOLOv9det)
            else:
                latter_data, latter_ellipse_axis, rects, boxes = bounding_para(latter_img, closing_kernel)
            l_diameters=calculate_diameter(latter_ellipse_axis)
            f_diameters =calculate_diameter(former_ellipse_axis)
            have_rain=any(l_diameters >=3) or any(f_diameters>=3)
            draw_data=[]
            if(SAVE_ARR_IMG and  have_rain ): #debug畫圖用
                r_latter_img = latter_img.copy()
                r_latter_img[:, :, 1 : 3] = 0
                compose_img = np.zeros((640, 480, 3), dtype=np.uint8)
                for i,(cy,cx) in enumerate(former_data[:,:2]): #data跟圖片是反的
                    s,l=former_ellipse_axis[i][0],former_ellipse_axis[i][1]
                    d=6.486e-2*((l**2*s)**(1 / 3))
                    class_index = np.digitize([d], rainclass)[0]
                    #if(class_index>=17):
                    x1 = max(0, int(cx- l / 2))
                    y1 = max(0, int(cy -l / 2))
                    x2 = min(480, int(cx + l / 2))
                    y2 = min(640, int(cy + l / 2))
                    compose_img[y1:y2,x1:x2]+=former_img[y1:y2,x1:x2]#[y1:y2, x1:x2]
                    draw_data.append([class_index,cy,cx])
                for i,(cy,cx) in enumerate(latter_data[:,:2]):
                    s,l=latter_ellipse_axis[i][0],latter_ellipse_axis[i][1]
                    d=6.486e-2*((l**2*s)**(1 / 3))
                    class_index = np.digitize([d], rainclass)[0]
                    #if(class_index>=17):
                    x1 = max(0, int(cx- l / 2))
                    y1 = max(0, int(cy -l / 2))
                    x2 = min(480, int(cx + l / 2))
                    y2 = min(640, int(cy + l / 2))
                    compose_img[y1:y2,x1:x2]+=r_latter_img[y1:y2,x1:x2]
                    draw_data.append([class_index,cy,cx])
                compose_img = cv2.transpose(compose_img)
                for row in draw_data:
                    if  abs(row[1] - 0) > abs(row[1]-640):
                        bias=-60
                    else:
                        bias=+5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(compose_img,str(row[0]),(int(row[1])+bias,int(row[2])+10),font,0.8, (255,255,0),3, cv2.LINE_AA)
                    cv2.putText(compose_img,"("+str(int(row[1]))+','+str(int(row[2]))+")",(int(row[1])+bias,int(row[2])+25),font,0.4, (255,255,0),2, cv2.LINE_AA)

                ax.imshow(compose_img)

            # get input data for ANN model
            input_data, input_data_, latter_len = make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES)      ## trace
            # if the image is not blank, then HAVE_DATA is true
            HAVE_DATA = True if input_data_.shape[0] != 0 else False
            # match raindrop pair using ANN model and return binary results
            coor_mask = coor_filter(input_data_)
            binary_y = match_raindrop(input_data, latter_len, sess, SIGMOID_THRESHOLD,input_data_
                                    ,rf_classifier,former_data,former_ellipse_axis,latter_data,latter_ellipse_axis,
                                    sess_y,X,k_prob,tst)
            
            for idx in range (coor_mask.shape[0]):
                binary_y[idx] = True if (binary_y[idx] == True and coor_mask[idx]==True) else False
            # calibrate results using preceding tracking path
            matched_pair= rainfall_agent.calibrating_path(input_data_,
                                                        binary_y,
                                                        (latter_data, latter_ellipse_axis),
                                                        compose_img_name,
                                                        NUM_FEATURES
                                                        )
            f_indices = find_matching_indices(former_data, matched_pair[:, :3])
            l_indices = find_matching_indices(latter_data, matched_pair[:, 3:6])
            matched_pairax= np.array([], dtype = np.float64).reshape(0, 11)
            for i in range(matched_pair.shape[0]):
                matched_pairax_=np.concatenate((matched_pair[i], former_ellipse_axis[f_indices[i]],latter_ellipse_axis[l_indices[i]]), axis=0)
                matched_pairax=np.vstack((matched_pairax,matched_pairax_))
            
            
            for i in range(matched_pair.shape[0]):
                vector = rainfall_agent .calculate_vector(matched_pair, i)
                latter_ax = rainfall_agent .find_elps_axis(matched_pair[i, 3 : -1], latter_data, latter_ellipse_axis)
                tracked_func(matched_pair[i, 0 : 3],matched_pair[i, 3 : -1],latter_ax, matched_pairax[i][7:9], vector,int(compose_img_name[22:25]),compose_img_name, ex_matched, first_new_raindrop,tracked_memory)
            ex_matched=matched_pairax.copy()
            ex_matched=np.append(ex_matched,np.full((ex_matched.shape[0],1),int(compose_img_name[22:25])),axis=1)
            if(SAVE_ARR_IMG and  have_rain ):
                draw_arrow(matched_pair, [], [], fig, ax)
                arrow_img_name = 'Arrow_' + compose_img_name
                fig.savefig( WRITE_PATH_ARRIMG + arrow_img_name, transparent = True, pad_inches = 0)
                fig.clear()

            img_pair_counter += 1
            if img_pair_counter % 100 == 0: print('{:d}, {:s}'.format(img_pair_counter, compose_img_name))
                                                                
            former_name         = latter_name
            former_img          = latter_img
            former_data         = latter_data
            former_ellipse_axis = latter_ellipse_axis
            if ((img_counter+2)==len(file_list)) :#last_img
                if(len(tracked_memory)>0):
                    output_memory.update(tracked_memory)
                    tracked_memory={}
            
        elif latter_name[0] == KEY_CHAR:
            # reset parameters
            if (int(latter_name[11 : 17]) != int(former_name[11 : 17]) or (img_counter+2)==len(file_list) ) :#last_img
                if(len(tracked_memory)>0):
                    output_memory.update(tracked_memory)
                    tracked_memory={}
            
            rainfall_agent.reset_memory()
            former_name = latter_name
            former_img_o=cv2.imread(READ_PATH + former_name)
            former_img = np.transpose(former_img_o, axes = (1, 0, 2))
            if USEYOLO==True :
                former_data, former_ellipse_axis=yolo_bounding(former_img_o,YOLOv9det)
            else:
                former_data, former_ellipse_axis, rects, boxes = bounding_para(former_img, closing_kernel)
            #former_ellipse_axis = np.zeros((1, 2))
        else:
            # give a fake name
            former_name = '9'*25
        img_counter += 1
        if ((img_counter % 500) == 0 ) :
            print("---------------------\n")
            print("img_counter = {}",img_counter)
            print("---------------------\n")
    processing_time = time.time()
    raindata=get_rain_data(output_memory, SHADOW_FIX, QC_OPERATION)
    record_raindrop_second(raindata,excel1)
    record_raindrop_min(raindata,excel3)
    record_raindrop_hour(excel3,excel2)
    csv_to_xlsx_pd(excel1, WRITE_PATH_ANALYZE)
    csv_to_xlsx_pd(excel2, WRITE_PATH_ANALYZE)
    csv_to_xlsx_pd(excel3, WRITE_PATH_ANALYZE)
    spand_time_0 = processing_time - start_time
    spand_time_hr  = spand_time_0 // 3600
    spand_time_min = (spand_time_0 % 3600) // 60
    spand_time_sec = spand_time_0 % 60
    diffThreshold      = int(5)
    ROI                = str("76 x 52")
    IS_modify          = True
    save_run_info(READ_PATH, start_time, processing_time, spand_time_0, diffThreshold, SAVE_ARR_IMG, ROI, IS_modify, (len(file_list)-1), WRITE_PATH )

    print("\n=========================================================================================================\n")
    print(READ_PATH)
    print("start time           :  ", time.ctime(start_time))
    print("modify               :  ", True                        )
    print("processing_end_time  :  ", time.ctime(processing_time) )
    print("total_seconds        :  ", spand_time_0, "sec"      )
    print("spend time           :  ", spand_time_hr,":",spand_time_min,":",spand_time_sec)
    print("\n=========================================================================================================\n") 

    # =========================================================================================================
    print("      _______   ______   .__   __.  _______")
    print("     |       \\ /  __  \\  |  \\ |  | |   ____|")
    print("     |  .--.  |  |  |  | |   \\|  | |  |__")
    print("     |  |  |  |  |  |  | |  . `  | |   __|")
    print("     |  '--'  |  `--'  | |  |\\   | |  |____")
    print("     |_______/ \\______/  |__| \\__| |_______|")

