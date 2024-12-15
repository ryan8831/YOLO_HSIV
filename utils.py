# @Author: jibowei
# @Date:   2019-04-25T16:16:33+08:00
# @Filenae: utils.py
# @Last modified by:   jibowei
# @Last modified time: 2019-04-25T16:16:33+08:00
# @License: MIT


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import cv2
import pandas as pd
from skimage import morphology
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import itertools
import csv
from numpy import matlib
from path_def import rainfall_measurement_path_def


WEIGHTS = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,],      #1為使用此資料 0為不使用資料 新版11個資料全部使用就填11個1 
                   dtype = np.bool_).reshape(1, -1)

### ========
# PATH DEFINE
REAULT_ID, READ_PATH, MODEL_READ_PATH, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, WRITE_PATH_MODIFY, KEY_CHAR = rainfall_measurement_path_def()


### ========

excel1='雨滴每秒結果'+ REAULT_ID
excel2='雨滴小時結果'+ REAULT_ID
excel3='雨滴分鐘結果'+ REAULT_ID


def rename_files(path):
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            parts = filename.split('_')
            last_part = parts[-1].split('.')[0]
            new_last_part = last_part.zfill(3)
            new_filename = '_'.join(parts[:-1]) + '_' + new_last_part + '.png'
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

def check_out_path_exist(check_path, id):
    if not os.path.exists(WRITE_PATH):
        os.makedirs(check_path)
        os.makedirs(check_path + 'analyze_' + id + '/')
        os.makedirs(check_path + 'ARR_ans_' + id + '/')
        os.makedirs(check_path + 'Modify_' + id + '/')

def csv_to_xlsx_pd(x, fpath):
    csv=pd.read_csv( fpath +'{}.csv'.format(x),encoding='utf-8')
    csv.to_excel( fpath +'{}.xlsx'.format(x),sheet_name='data')

def del_ds_store(file_list):
    if file_list[0] == '.DS_Store': file_list.remove('.DS_Store')

def set_ax(ax, position):
    ax.set_position(position)
    ax.set_axis_off()
    return ax

def fnc_show_img(img, gray):
    img = np.asarray(img)
    if gray == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def bounding_para(latter_img, closing_kernel):
    output = np.array([], dtype = np.float64).reshape(0, 3)
    ellipse_axis = np.array([], dtype = np.float64).reshape(0, 2)
    rects, boxes = [], []
    img = morphology.closing(latter_img[:, :, 0], closing_kernel) # https://jason-chen-1992.weebly.com/home/-morphology
    # 接著把Closing 拿來跟 Opening 比你會發現，其實就是反過來做而已。
    # 變成先Dilation 再 Erosion，等同你拿Structures Element 在Target 的外部移動，
    # 進不去的地方就把它填滿，可以將圖形內陷的銳角給鈍化。

    #show_image = np.asarray(latter_img[:, :, 1])
    #plt.imshow(show_image, cmap='gray')
    #plt.show()
    edge_img = canny(img)
    edge_img = np.ceil(edge_img >= 0.5).astype(int)
    #label = morphology.label(edge_img, neighbors = 8, return_num = True, connectivity = 2)
    label = morphology.label(edge_img, return_num = True, connectivity = 2)
    #show_image = np.asarray(edge_img)
    #plt.imshow(show_image)
    #plt.show()

    for i in range(1, (label[1] + 1)):
        points = np.argwhere(label[0] == i)
        rect = cv2.minAreaRect(points)         # rect：（最小外接矩形的中心（x，y），（寬度，高度），旋轉角度）
                                               # https://blog.csdn.net/CPZ742954007/article/details/81296331
        rects.append(rect)
        box = np.float32(cv2.boxPoints(rect))  # 算出四個頂點
        boxes.append(box)
        #if a, b != 0, area = ellipse area else num_pixels
        if rect[1][0]*rect[1][1] != 0:
            area = np.pi*rect[1][0]*rect[1][1] / 4      # 橢圓面積
            elps_ax = np.array(rect[1]).reshape(1, -1)  # https://blog.csdn.net/W_weiying/article/details/82112337
            elps_ax.sort()    # @@?
        else:  # @@?
            area = points.shape[0]
            elps_ax = np.repeat(2*np.sqrt(area / np.pi), 2).reshape(1, -1)    #@@?
        rect_ = np.array([rect[0][0], rect[0][1], area]).reshape(1, 3)
        output = np.vstack((output, rect_))                                   # 垂方向堆疊
        ellipse_axis = np.vstack((ellipse_axis, elps_ax))
        
    return output, ellipse_axis, rects, boxes

def make_input_data(former_data, latter_data, normalize_para, NUM_FEATURES, epsilon = 1e-12):
    output = np.array([], dtype = np.float64).reshape(0, (NUM_FEATURES-1))
    former_len = former_data.shape[0]
    latter_len = latter_data.shape[0]  #
    if former_len != 0 and latter_len != 0:
        sub_data = latter_data[:, 0 : 3].reshape(-1, 3)    # @@? why reshape
        for i in range(former_len):
            main_data = np.matlib.repmat(former_data[i][0 : 3].reshape(1, 3), latter_len, 1)
            # print( np.hstack((main_data, sub_data)).shape)
            # print(output.shape)
            output = np.vstack((output, np.hstack((main_data, sub_data))))#前後兩張圖雨滴的排列組合
    velocity = (8.13e-5)*np.sqrt((output[:, 0] - output[:, 3])**2 + (output[:, 1] - output[:, 4])**2)*500    # @@? 8.13e-5 ??
    output = np.hstack((output, velocity.reshape(output.shape[0], 1)))
    output_ = output.copy()  #有_就是原始圖
    output = (output - normalize_para[1][WEIGHTS]) / np.sqrt(normalize_para[0][WEIGHTS] + epsilon)           # @@?
    output /= normalize_para[2][WEIGHTS.squeeze()] # 從數組的形狀中刪除單維度條目，即把shape中為1的維度去掉      # @@?
    # print(output)
    return output, output_, latter_len

def draw_arrow(matched_pair, rects, boxes, fig, ax, mode = 'default'):
    if matched_pair.shape[0] != 0:
        num_arrow = matched_pair.shape[0]
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        for i in range(num_arrow):
            pair = matched_pair[i]
            dx = pair[3] - pair[0]
            dy = pair[4] - pair[1]
            ax.arrow(pair[0], pair[1],
                     dx, dy,
                     width = 1e-5,
                     linewidth = 0.5,
                     head_width = 1.,
                     head_length = 1.5,
                     figure = fig,
                     color = (0, 1, 0)
                     )
        if mode == 'default':
            color_ = 'white' if mode == 'f' else 'red'
            for rect in rects:
                p = patches.Ellipse(rect[0],
                                    rect[1][0], rect[1][1],
                                    angle = rect[2],
                                    fill = True,
                                    color = color_,
                                    edgecolor = color_,
                                    linewidth = 0.1)
                ax.add_patch(p)
            for box in boxes:
                box = np.vstack((box, box[0, :].reshape(1, -1)))
                verts = box.tolist()
                path = Path(verts, codes)
                p = patches.PathPatch(path,
                                      fill = False,
                                      edgecolor = 'yellow',
                                      linewidth = 0.1)
                ax.add_patch(p)

def refine_file_list(file_list, remain_target): #????????
    output = []
    for i in range(len(file_list)):
        for j in remain_target:
            if file_list[i][11 : 13] == j:
                output.append(file_list[i])
    return output

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = 'Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# rainfall meter================================================================
class Rainfall_Meter:
    def __init__(self, accept_dis, ana_path, max_ex_matched = 75):
        self.accept_dis = accept_dis
        self.ana_path = ana_path
        self.max_ex_matched = max_ex_matched
        self.memory = {}
        self.new_memory = {}
        self.full_memory = {}
        self.matched_memory = {}
        self.matched_memory[0] = 0
        self.counted_raindrop = {}
        self.rainfall_minute = {}
        self.rainfall_hour = {}
        self.rainfall = 0
        self.num_raindrop = 0
        self.compose_img_name_ = 0

    def calibrating_path(self, input_data_, binary_y, latter_data, compose_img_name, NUM_FEATURES):
        if input_data_.shape[0] != 0:
            latter_elps_ax = latter_data[1] # 長短軸
            latter_data = latter_data[0]    # x,y,area
            binary_y = np.matlib.repmat(binary_y, 1, NUM_FEATURES)  # binray是雨滴配對完的結果
            matched_pair = input_data_[binary_y].reshape(-1, NUM_FEATURES) # 把true的資料保留下來(x,y,area,速度)
            ex_matched_pair = np.zeros((self.max_ex_matched, 7))           # @@?
            #deal with ex_match
            i = 0
            for key in list(self.memory.keys()):          # 會新增但不再進迴圈
                pre_vector = self.memory[key][0]          # 假設現在是圖片223 prevector=222-221
                coor = self.memory[key][1].reshape(1, -1) # key[1]=former_data=圖片222的座標,面積
                distance, distance_mask = self.calculate_distance(coor, pre_vector, latter_data) # distance:實際座標與預估座標的距離
                if np.sum(distance_mask.astype(int)) != 0:    # if we have proper choice
                    ex_matched_pair[i, : 3] = self.memory[key][1]    # 與262行相同
                    ex_matched_pair[i, 3 : -1] = latter_data[distance_mask].reshape(1, -1)# 3:-1 把3到最後一個前的元素都給替換掉
                    vector = self.calculate_vector(ex_matched_pair, i)
                    self.creat_new_record(latter_data[distance_mask], latter_elps_ax[distance_mask], vector)
                    i = i + 1
            ex_matched_pair = self.remove_zero(ex_matched_pair, mode = 'uncomplete', need_sub = 1)    # for 迴圈執行完將雨滴資料紀錄後再一次將所有0去除
            # "-1"用以表示當前維度所有子模塊最後一個，"負號用以表示從後往前數的元素,-n即是表示從後往前數的第n個元素"
            ex_matched_pair[:, -1] = self.calculate_velocity(ex_matched_pair, default_ = True)
            #deal with match
            for j in range(matched_pair.shape[0]):        # matched_pair : fx, fy, farea, lx, ly, larea, velocity
                former_target = matched_pair[j, : 3].reshape(1, -1)   #前一張第j個配對到的雨滴data
                if self.trans(former_target) in self.memory.keys(): #這組雨滴data是否存在前一次配對過的memory
                    pre_vector = (self.memory[self.trans(former_target)])[0]
                    coor = former_target.reshape(1, -1)
                    distance, distance_mask = self.calculate_distance(coor, pre_vector, latter_data)
                    if np.sum(distance_mask.astype(int)) != 0:    # if we have proper choice
                        matched_pair[j, 3 : -1] = latter_data[distance_mask].reshape(1, -1)
                        vector = self.calculate_vector(matched_pair, j)
                        self.creat_new_record(latter_data[distance_mask], latter_elps_ax[distance_mask], vector)
                        self.remove_old_record(former_target)
                    else:
                        vector = self.calculate_vector(matched_pair, j)
                        ax = self.find_elps_axis(matched_pair[j, 3 : -1], latter_data, latter_elps_ax)
                        self.creat_new_record(matched_pair[j, 3 : -1], ax, vector)
                        self.remove_old_record(former_target)
                        matched_pair[j, :] = 0
                else:    # means we have new raindrop
                    vector = self.calculate_vector(matched_pair, j)
                    ax = self.find_elps_axis(matched_pair[j, 3 : -1], latter_data, latter_elps_ax)#藉由配對到的雨滴與資料找回長短軸資訊
                    self.creat_new_record(matched_pair[j, 3 : -1], ax, vector)
            matched_pair = self.process_matched_pair(matched_pair, ex_matched_pair, NUM_FEATURES)
            self.matched_memory[compose_img_name] = matched_pair  # 存前後影像雨滴配對資訊(前一顆與下一顆為同一列)
            self.full_memory[compose_img_name] = self.memory    # 存有被配對到的雨滴(只有該顆雨滴的向量、座標、面積、長短軸)
            self.record_new_raindrop_info(compose_img_name)
            self.refresh_memory()
            return matched_pair
        else:
            self.matched_memory[compose_img_name] = np.array([]).reshape(0, 7)
            self.compose_img_name_ = compose_img_name
            self.reset_memory()
            return np.array([]).reshape(0, NUM_FEATURES)

    def calculate_diameter(self, target):
        return (8.13e-2)*(((target[:, 4]**2)*target[:, 3])**(1 / 3))     # Deq = (6V/pi)^(1/3) = (l^2 * s)^(1/3)   # from papper : HSIV System for Rainfall Measurement

    def calculate_distance(self, coor, pre_vector, latter_data):
        assumed_coor = coor[0, : 2].reshape(1, -1) + pre_vector
        distance = np.sqrt(np.sum((assumed_coor - latter_data[:, : 2])**2, axis = 1))
        distance_mask = (distance <= self.accept_dis)
        if np.sum(distance_mask) > 1:   #coor=現在的雨滴 ,-1為面積
            min_area_diff = np.argmin(np.abs(coor[0, -1] - latter_data[distance_mask, -1]))#np.argmin()返回最小值的位子
            distance_mask = np.zeros((distance.shape))
            distance_mask[min_area_diff] = 1 #把argmin找出來最小的那個面積 把它設成1
            distance_mask = distance_mask.astype(bool)
        return distance, distance_mask

    def calculate_vector(self, matched_pair, i):
        return (matched_pair[i, 3 : 5] - matched_pair[i, 0 : 2]).reshape(1, -1)

    def calculate_velocity(self, matched_pair, default_):
        if default_:
            x_diff = matched_pair[:, 0] - matched_pair[:, 3]
            y_diff = matched_pair[:, 1] - matched_pair[:, 4]
        else:
            x_diff = matched_pair[:, 5]
            y_diff = matched_pair[:, 6]
        return (8.13e-5)*np.sqrt((x_diff)**2 + (y_diff)**2)*500

    def creat_new_record(self, target, ax, vector):  # target = latter_data
        new_target = copy.deepcopy(target)
        new_target = new_target.reshape(1, -1)
        if ax.shape[1] > 2: ax = ax[0, : 2]       #ax:長短軸  #防呆
        ax = ax.reshape(1, -1)
        key = self.trans(new_target)    # new_target轉成 list再轉乘 str ，做為memory的key
        if key not in self.memory: self.memory[key] = (vector, new_target, ax)
        if key not in self.new_memory: self.new_memory[key] = (vector, new_target, ax)

    def find_elps_axis(self, target, latter_data, latter_elps_ax):
        target = target.reshape(1, -1)
        key = (latter_data == target)[:, : 2]       #bool
        return latter_elps_ax[key].reshape(1, -1)

    def find_time(self, time, mode):
        output = []
        if mode == 'MIN':
            for key in list(self.full_memory.keys()):
                if key[11 : -12] == time: output.append(self.full_memory[key])
        elif mode == 'HR':
            for key in list(self.rainfall_minute.keys()):
                if key[: -2] == time: output.append(self.rainfall_minute[key])
        return output

    def out_tracking(self, target, matched_p):
        result_1 = matched_p[:, 0] == target[0, 0]
        result_2 = matched_p[:, 1] == target[0, 1]
        result_3 = matched_p[:, 2] == target[0, 2]
        result = result_1*result_2*result_3
        return True if np.sum(result, axis = None) == 0 else False

    def process_matched_pair(self, matched_pair, ex_matched_pair, NUM_FEATURES):
        matched_pair[:, -1] = self.calculate_velocity(matched_pair, default_ = True)
        matched_pair = np.vstack((matched_pair, ex_matched_pair))       
        if matched_pair.shape[0] != 0: matched_pair = np.unique(matched_pair, axis = 0)
        return self.remove_zero(matched_pair, mode = 'complete')

    def record_rainfall_in_minute(self, compose_img_name):
        minute = compose_img_name[11 : -14]
        data = self.counted_raindrop[compose_img_name]
        if data.shape[0] != 0:
            short_axis = (8.13e-2)*data[:, 3]
            long_axis = (8.13e-2)*data[:, 4]
            tmp_rainfall = (np.pi / 6)*np.sum((long_axis**2)*short_axis)     # @@?
            tmp_rainfall = (tmp_rainfall*30) / (120*39)
            tmp_num_raindrop = data.shape[0]*30                              # @@?
            self.rainfall += tmp_rainfall
            self.num_raindrop += tmp_num_raindrop
            if minute not in self.rainfall_minute: self.rainfall_minute[minute] = [0, 0, 0, 0]
            self.rainfall_minute[minute][0] += tmp_rainfall
            self.rainfall_minute[minute][1] += tmp_num_raindrop
            self.rainfall_minute[minute][2] = self.rainfall
            self.rainfall_minute[minute][3] = self.num_raindrop
            # print(minute)
            # print(compose_img_name)
            with open( self.ana_path + '{}.csv'.format(excel3),'a',newline='', encoding='UTF-8') as f9:
                    writer=csv.writer(f9)
                    writer.writerow([minute,self.rainfall_minute[minute][0],self.rainfall_minute[minute][1],self.rainfall_minute[minute][2],self.rainfall_minute[minute][3]])



    def record_rainfall_in_hour(self, compose_img_name):
        print('Hi')
        hour = compose_img_name[11 : -16]
        tmp_rainfall = 0
        tmp_num_raindrop = 0
        hour_list = self.find_time(hour, mode = 'HR')
        print(hour_list)     #此分鐘總雨量 此分鐘數量 總雨量 總數量
        for data in hour_list:
            tmp_rainfall += data[0]
            tmp_num_raindrop += data[1]
        self.rainfall_hour[hour] = [tmp_rainfall, tmp_num_raindrop, self.rainfall, self.num_raindrop]  #這個小時總雨量 這個小時總數量 總雨量 總數量
        print(hour)
        with open( self.ana_path +'{}.csv'.format(excel2),'a',newline='', encoding='UTF-8') as f8:
                    writer=csv.writer(f8)
                    writer.writerow([hour,self.rainfall_hour[hour][0],self.rainfall_hour[hour][1],self.rainfall_hour[hour][2],self.rainfall_hour[hour][3]])

    def record_new_raindrop_info(self, compose_img_name): #200.175
        tmp_memory = np.zeros((self.max_ex_matched, 9))     # max_ex_matched: 一張圖中最多配對75組  
        matched_p_ = self.matched_memory[self.compose_img_name_]        # self.compose_img_name_ : 前一次配對的，例如現在是222_223，compose_img_name_就是221_222
        matched_p = self.matched_memory[compose_img_name]               # self.matched_memory : 前後影像雨滴配對資訊(前一顆與下一顆為同一列)
        if self.compose_img_name_ != 0 and matched_p_.shape[0] != 0:
            memory_ = self.full_memory[self.compose_img_name_]         # self.full_memory :被配對到的雨滴(只有該顆雨滴的向量、座標、面積、長短軸)
            # memory_ : 前一次compose_img_name的紅色
            for i in range(matched_p_.shape[0]):
                key = self.trans(matched_p_[i, 3 : -1].reshape(1, -1))
                if self.out_tracking(matched_p_[i, 3 : -1].reshape(1, -1), matched_p):    # 輸出追蹤 偵測出前一次被配對的雨滴是否出現在這次的配對中(前一次的紅色是否與這次有配對到的白色相同)
                                                                                          # true代表前一張被配對的雨滴沒出現在這次的配對中，
                                                                                          # 也代表了前一張被配對的雨滴是最後一次出現在這分鐘內，
                                                                                          # 這時才算開顆雨滴追蹤結束，並將它紀錄起來
                    new_raindrop = copy.deepcopy(memory_[key])
                    tmp_memory[i, : 7] = np.hstack((new_raindrop[1], new_raindrop[2], new_raindrop[0]))
            tmp_memory = self.remove_zero(tmp_memory, mode = 'uncomplete', need_sub = 2)
            tmp_memory[:, -2] = self.calculate_velocity(tmp_memory, default_ = False)
            tmp_memory[:, -1] = self.calculate_diameter(tmp_memory)
            tmp_memory = self.remove_bugs(tmp_memory)
            if tmp_memory.shape[0] != 0:
                self.counted_raindrop[compose_img_name] = tmp_memory
                self.record_rainfall_in_minute(compose_img_name)
        self.compose_img_name_ = compose_img_name

    def record_last_img_raindrop_info(self, compose_img_name, HAVE_DATA):
        if HAVE_DATA:
            #print('Last_HI')
            tmp_memory = np.zeros((self.max_ex_matched, 9))
            matched_p_ = self.matched_memory[self.compose_img_name_]       # 這裡的self.compose_img_name_已經變成跟原self.compose_img_name相同
            memory_ = self.full_memory[self.compose_img_name_]
            for i in range(matched_p_.shape[0]):
                key = self.trans(matched_p_[i, 3 : -1].reshape(1, -1))
                new_raindrop = memory_[key]
                #print('key = ', memory_[key] )
                #print('new_raindrop[1]={}'.format(new_raindrop[1]))    #雨滴位置與面積
                #print('new_raindrop[2]={}'.format(new_raindrop[2]))    #短軸、長軸
                #print('new_raindrop[0]={}'.format(new_raindrop[0]))    #與上一張圖片X1 Y1之差值(向量)
                tmp_memory[i, : 7] = np.hstack((new_raindrop[1], new_raindrop[2], new_raindrop[0]))
            tmp_memory = self.remove_zero(tmp_memory, mode = 'uncomplete', need_sub = 2)
            tmp_memory[:, -2] = self.calculate_velocity(tmp_memory, default_ = False)
            tmp_memory[:, -1] = self.calculate_diameter(tmp_memory)
            tmp_memory = self.remove_bugs(tmp_memory)
            if tmp_memory.shape[0] != 0:
                self.counted_raindrop[compose_img_name] = tmp_memory
                self.record_rainfall_in_minute(compose_img_name)
            # print(len(tmp_memory))
            # print(compose_img_name)
            for i in range(0,len(tmp_memory)):
                with open( self.ana_path + '{}.csv'.format(excel1),'a',newline='', encoding='UTF-8') as f7:
                    writer=csv.writer(f7)
                    writer.writerow([compose_img_name,tmp_memory[i][0],tmp_memory[i][1],tmp_memory[i][2],tmp_memory[i][3],tmp_memory[i][4],tmp_memory[i][5],tmp_memory[i][6],tmp_memory[i][7],tmp_memory[i][8]])
            # print(self.counted_raindrop)
            # 輸出為九個數字分別代表此柯雨滴最後位置 1.X1 2.Y1 3.area 4.長短軸 5.長短軸 6.向量 7.向量 8.速度 9.直徑
        self.compose_img_name_ = 0

    def remove_bugs(self, target):
        target_col = target.shape[1]
        nonbug = target[:, -1] >= 0.15   # 粒徑要大於0.15
        nonbug = np.matlib.repmat(nonbug.reshape(-1, 1), 1, target.shape[1])
        return target[nonbug].reshape(-1, target_col)

    def remove_zero(self, target, mode, need_sub = 0):
        col = target.shape[1]  # 75*7 雨滴配對max
        nonzeros = target != 0
        if mode == 'complete':
            nonzeros = np.sum(nonzeros, axis = 1) == col
        elif mode == 'uncomplete':
            nonzeros = np.sum(nonzeros, axis = 1) == (col - need_sub)
        nonzeros = np.matlib.repmat(nonzeros.reshape(-1, 1), 1, col)
        return target[nonzeros].reshape(-1, col)

    def remove_old_record(self, target):
        del self.memory[self.trans(target)]

    def refresh_memory(self):
        self.memory = copy.deepcopy(self.new_memory)
        self.new_memory = {}

    def reset_memory(self):
        self.memory = {}
        self.new_memory = {}

    def trans(self, target):
        return str(target.tolist())
