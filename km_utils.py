#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import csv
import copy
import time
import shutil
import itertools
import numpy as np
import pandas as pd
from skimage         import morphology
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL             import Image, ImageDraw, ImageFont
from numpy           import matlib
from path_def        import rainfall_measurement_path_def
from scipy.optimize  import linear_sum_assignment
from matplotlib.path import Path

WEIGHTS = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,],      #1為使用此資料 0為不使用資料 新版11個資料全部使用就填11個1 
                   dtype = np.bool_).reshape(1, -1)

### ========
# PATH DEFINE
RESULT_ID, READ_PATH, MODEL_READ_PATH, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, WRITE_PATH_MODIFY, KEY_CHAR, SAVE_ARR_IMG, SHADOW_FIX, QC_OPERATION,FORESTER_MODEL_PATH = rainfall_measurement_path_def()

### ========
excel1='雨滴每秒結果'+ RESULT_ID
excel2='雨滴小時結果'+ RESULT_ID
excel3='雨滴分鐘結果'+ RESULT_ID

def reconnecting_cost_matrix(target, key, latter_frame, vector, tracked_memory):
    #預測位置
    pre_x = ( latter_frame - tracked_memory[key].frames ) * tracked_memory[key].vector[0][0] + tracked_memory[key].boundingbox[0][0] #(現在frame- 有追蹤到的frame) * vector +座標
    pre_y = ( latter_frame -  tracked_memory[key].frames ) * tracked_memory[key].vector[0][1] + tracked_memory[key].boundingbox[0][1]
    pre_position = np.array([np.array([pre_x,pre_y])])
    distances = np.sqrt(((pre_position[np.newaxis, :, :2] - target[np.newaxis, :, :2])**2).sum(axis=2))
    #速度
    pre_velocity = np.sqrt(abs(tracked_memory[key].vector[0][0])**2 + abs(tracked_memory[key].vector[0][1])**2)
    latter_velocity =np.sqrt(abs(vector[0][0])**2 + (vector[0][1])**2)
    velocity_diff = abs(pre_velocity-latter_velocity)
    #面積
    pre_area =  np.array([np.array([tracked_memory[key].boundingbox[0][2]])])
    area_diff = np.abs(pre_area - target[np.newaxis, :, 2])
    #角度
    prevector_deg = np.rad2deg(np.arctan2(-tracked_memory[key].vector[0][1],tracked_memory[key].vector[0][0]))
    prevector_deg=np.where(prevector_deg<0,prevector_deg+360,prevector_deg)
    deg=np.rad2deg(np.arctan2(-vector[0][1], vector[0][0]))
    deg=np.where(deg<0,deg+360,deg)
    angle_diff = np.abs(prevector_deg - deg)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)  #維度注意
    n_distances=distances/800*100
    n_angle_diff=angle_diff/180*100
    n_area_diff=area_diff
    cost_matrix = n_distances*0.4 + n_angle_diff*0.3 + n_area_diff*0.1+velocity_diff*0.2
    return cost_matrix

def determine_reconnecting(former, latter, former_ax, vector, ax, latter_frame, compose_img_name,tracked_memory ):  
    repair_raindrop =0
    for key in tracked_memory.keys():
        target=latter.reshape(1,-1)
        if((tracked_memory[key].steps >0) and (tracked_memory[key].steps>=(latter_frame-tracked_memory[key].frames))): #steps
            cost_matrix = reconnecting_cost_matrix(target, key, latter_frame, vector, tracked_memory) 
            pre_y = ( latter_frame -  tracked_memory[key].frames ) * tracked_memory[key].vector[0][1] + tracked_memory[key].boundingbox[0][1]
            if(cost_matrix[0]<5 and  former[1]>(pre_y-2*abs(tracked_memory[key].vector[0][1]))):    # 代表為斷掉重連，需更新
                repair_raindrop+=1
                steps  = calculate_frame(target[0][:2], vector)
                update_target= copy.deepcopy(target).reshape(1,-1)      
                angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
                last_angle = np.rad2deg(np.arctan2(-(tracked_memory[key].vector[0][1]), tracked_memory[key].vector[0][0])) +90
                delta_angle =abs(angle- last_angle)
                #取最大長短軸
                if(tracked_memory[key].boundingbox[0][2]>update_target[0][2] or tracked_memory[key].boundingbox[0][2]==update_target[0][2] ):
                    update_ax=tracked_memory[key].ax
                else:
                    update_ax=ax.reshape(1,-1)
                
                #取最大角度及最大角度變化量
                if(angle>0):
                    update_angle = max(angle, tracked_memory[key].angle)
                else:
                    update_angle= min(angle, tracked_memory[key].angle)
                
                update_maxdelta_angle = max(delta_angle, tracked_memory[key].maxdelta_angle)
                new_value = trackers(vector, update_target, update_ax, latter_frame, update_angle, update_maxdelta_angle, steps, compose_img_name)
                tracked_memory[key] =new_value
    if(repair_raindrop==0 and latter[2]>145): #沒救回來且直徑大於等於class9
        steps  = calculate_frame(latter[0:2], vector)
        angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
        delta_angle= 0
        if(former[2]>latter[2]):
            max_ax =former_ax.reshape(1,-1)
        else:
            max_ax= ax.reshape(1,-1)
        create_tracking_raindrop(vector, latter, max_ax, latter_frame, angle, delta_angle, steps, compose_img_name,tracked_memory )
    


def calculate_frame(target, vector):    
    #左右step
    if vector[0][0] > 0:
        steps_x = (640 - target[0]) // vector[0][0]
    elif vector[0][0] < 0:
        steps_x = (0 - target[0]) // vector[0][0]
    else:
        steps_x = float('inf')   
    # 下step
    if vector[0][1] > 0:
        steps_y = (480 - target[1]) // vector[0][1]
    else:
        steps_y = float('inf')  
    min_steps= min(steps_x, steps_y) #最小step
    return  min_steps

def tracked_func(former, latter, ax,  former_ax,vector, latter_frame, compose_img_name, ex_tracked, first_new_raindrop, tracked_memory): #former、latter是一維，vector二維
    if(len(ex_tracked)>0):
        if((latter_frame-int(ex_tracked[0][-1]))==1):
            check_interrupt = 0
            for i in range(len(ex_tracked)):
                comparison_result =(former == ex_tracked[i,3:6])   # 這張配對的former== 上一張配對的latter(比較皆是一維)
                if np.sum(comparison_result, axis=None) == 3: #即追蹤成功，往下檢查是否更新
                    check_interrupt += 1
                    if(len(tracked_memory)>0):
                        is_update =0 
                        for key in tracked_memory.keys():
                            comparison_update_result = (former ==tracked_memory[key].boundingbox[0] )
                            if np.sum(comparison_update_result, axis=None) == 3: #需更新
                                steps  = calculate_frame(latter[0:2], vector)
                                update_target= copy.deepcopy(latter).reshape(1,-1)
                                angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
                                last_angle = np.rad2deg(np.arctan2(-(tracked_memory[key].vector[0][1]), tracked_memory[key].vector[0][0])) +90
                                delta_angle =abs(angle- last_angle)
                                #取最大長短軸
                                if(tracked_memory[key].boundingbox[0][2]>update_target[0][2] or tracked_memory[key].boundingbox[0][2]==update_target[0][2] ):
                                    update_ax=tracked_memory[key].ax
                                else:
                                    update_ax=ax.reshape(1,-1)
                                
                                #取最大角度及最大角度變化量
                                if(angle>0):
                                    update_angle = max(angle, tracked_memory[key].angle)
                                else:
                                    update_angle= min(angle, tracked_memory[key].angle)
                                
                                update_maxdelta_angle = max(delta_angle, tracked_memory[key].maxdelta_angle)
                                new_value = trackers(vector, update_target, update_ax, latter_frame, update_angle, update_maxdelta_angle, steps, compose_img_name)
                                tracked_memory[key] =new_value
                                is_update +=1
                        if(is_update==0): #沒更新則代表是新追蹤到雨滴，避免重複寫入
                            steps  = calculate_frame(latter[0:2], vector)
                            angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
                            delta_angle= 0
                            #取最大長短軸
                            if(ex_tracked[i][2]>ex_tracked[i][5]):
                                max_ax = ex_tracked[i][7:9].reshape(1,-1)
                            else:
                                max_ax = ex_tracked[i][9:11].reshape(1,-1)
                            if(latter[2]>max(ex_tracked[i][2], ex_tracked[i][5])):
                                max_ax =ax.reshape(1,-1)
                            create_tracking_raindrop(vector, latter, max_ax, latter_frame, angle, delta_angle, steps, compose_img_name,tracked_memory )
                    else:   #tracked_mem沒東西又配對成功，即新追蹤雨滴
                        steps  = calculate_frame(latter[0:2], vector)
                        angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
                        delta_angle= 0
                        if(ex_tracked[i][2]>ex_tracked[i][5]):
                            max_ax = ex_tracked[i][7:9].reshape(1,-1)
                        else:
                            max_ax = ex_tracked[i][9:11].reshape(1,-1)
                        if(latter[2]>max(ex_tracked[i][2], ex_tracked[i][5])):
                            max_ax =ax.reshape(1,-1)
                        create_tracking_raindrop(vector, latter, max_ax, latter_frame, angle, delta_angle, steps, compose_img_name,tracked_memory )
            if(check_interrupt==0): #開始檢查是否為斷掉重連
                determine_reconnecting(former, latter, former_ax, vector, ax, latter_frame, compose_img_name,tracked_memory )
        
        else: #突然出現的第1條線
            determine_reconnecting(former, latter, former_ax, vector, ax, latter_frame, compose_img_name,tracked_memory )
        
    else: # 影像斷掉後第一張
        if(len(tracked_memory)>0):
            determine_reconnecting(former, latter, former_ax, vector, ax, latter_frame, compose_img_name,tracked_memory )
        else:
             if(former[2]>145 or latter[2]>145): #沒救回來且直徑大於class9
                    steps  = calculate_frame(latter[0:2], vector)
                    angle= np.rad2deg(np.arctan2(-vector[0][1], vector[0][0])) +90
                    delta_angle= 0
                    if(former[2]>latter[2]):
                        max_ax =former_ax.reshape(1,-1)
                    else:
                        max_ax= ax.reshape(1,-1)
                    create_tracking_raindrop(vector, latter, max_ax, latter_frame, angle, delta_angle, steps, compose_img_name,tracked_memory )


def create_tracking_raindrop(vector, target, ax,  first_frame, angle, maxdelta_angle, steps, compose_img_name,tracked_memory):
    new_target = copy.deepcopy(target)
    new_target = new_target.reshape(1, -1)
    if ax.shape[1] > 2: ax = ax[0, : 2]       #ax:長短軸  #防呆
    ax = ax.reshape(1, -1)
    key =str(new_target.tolist())
    value=trackers(vector, new_target, ax, first_frame, angle, maxdelta_angle, steps, compose_img_name)
    tracked_memory[key] = value


def normalize_to_percentage(matrix,type):
    min_value = 0
    max_value={'dis':400,'deg':360,'area':100}
    max_value=max_value[type]
    return (matrix - min_value) / (max_value - min_value)*100
def calculate_cost(prediction,latter_data,former_positions,dis_w,deg_w,area_w):
    distances = np.sqrt(((prediction[:, np.newaxis, :2] - latter_data[np.newaxis, :, :2])**2).sum(axis=2))
    area_diff = np.abs(prediction[:, np.newaxis, 2] - latter_data[np.newaxis, :, 2])
    x_diff=prediction[:,0]-former_positions[:,0]
    y_diff=prediction[:,1]-former_positions[:,1]
    prevector_deg = np.rad2deg(np.arctan2(-y_diff, x_diff))
    prevector_deg=np.where(prevector_deg<0,prevector_deg+360,prevector_deg)
    x_diff = latter_data[:, 0] - former_positions[:, 0, np.newaxis]
    y_diff = latter_data[:, 1] - former_positions[:, 1, np.newaxis]
    deg=np.rad2deg(np.arctan2(-y_diff, x_diff))
    deg=np.where(deg<0,deg+360,deg)
    angle_diff = np.abs(prevector_deg[:, np.newaxis] - deg)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
    n_distances=normalize_to_percentage(distances,'dis')
    n_angle_diff=normalize_to_percentage(angle_diff,'deg')
    n_area_diff=normalize_to_percentage(area_diff,'area')
    cost_matrix = n_distances*dis_w + n_angle_diff*deg_w + n_area_diff*area_w
    return cost_matrix


def hungarian_algorithm(prediction,latter_data,former_positions,dis_w,deg_w,area_w):
    cost_matrix = calculate_cost(prediction,latter_data,former_positions,dis_w,deg_w,area_w)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix[row_ind, col_ind]
def mem_to_array(mem):
    predictions=[]
    former_positions = []
    for values in mem.values():
        former_positions.append(values[1].copy())
        prediction=values[1].copy()
        prediction[:, :2] += values[0].reshape(-1)
        predictions.append(prediction)
    former_positions=np.concatenate(former_positions, axis=0,dtype=np.float64)
    predictions=np.concatenate(predictions, axis=0,dtype=np.float64)
    return former_positions,predictions
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
    # # # #label = morphology.label(edge_img, neighbors = 8, return_num = True, connectivity = 2)
    label = morphology.label(edge_img, return_num = True, connectivity = 2)

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
    return output, output_, latter_len    # output是為了給神經網路跑模型

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

def refine_file_list(file_list, remain_target):
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
class trackers:
    def __init__(self, vector, boundingbox, ax, frames, angle, maxdelta_angle,  steps=0, compose_img_name=" "):
        self.vector=vector
        self.boundingbox=boundingbox
        self.ax=ax
        self.frames=frames
        self.steps=steps
        self.compose_img_name=compose_img_name
        self.first_new_raindrop = {}
        self.tracked_memory = {}
        self.angle = angle
        self.maxdelta_angle = maxdelta_angle
        
# rainfall meter================================================================
class Rainfall_Meter:
    def __init__(self,ana_path, max_ex_matched = 75):
        self.ana_path = ana_path
        self.max_ex_matched = max_ex_matched
        self.memory = {}
        self.new_memory = {}
        self.full_memory = {}
        self.matched_memory={}
        self.matched_memory[0] = 0  # 保存每連續兩偵有match的雨滴資訊，索引名稱為該次的"compose_img_name"，若無match或該次沒有偵測到雨滴則該索引中為[]
        self.counted_raindrop = {}
        self.rainfall_minute = {}
        self.rainfall_hour = {}
        self.rainfall = 0
        self.num_raindrop = 0
        self.compose_img_name_ = 0
        self.costdata=[]

    def calibrating_path(self, input_data_, binary_y, latter_data, compose_img_name, NUM_FEATURES):
        if input_data_.shape[0] != 0:
            latter_elps_ax = latter_data[1] # 長短軸
            latter_data = latter_data[0]    # x,y,area
            binary_y = np.matlib.repmat(binary_y, 1, NUM_FEATURES)         # binray是雨滴配對完的結果
            matched_pair = input_data_[binary_y].reshape(-1, NUM_FEATURES) # 把true的資料保留下來(x,y,area,速度)
            #deal with ex_match                                            # 前一張的02配對現在這張03
            ex_matched_pair=[]
            if len(self.memory) > 0:
                former_positions,prediction=mem_to_array(self.memory)
                row_indices, col_indices, costs = hungarian_algorithm (prediction,latter_data,former_positions,dis_w=0.5,deg_w=0.3,area_w=0.2)
                for row, col, cost in zip(row_indices, col_indices, costs):
                    cost_range=15 #20 #8+former_positions[row,2]/50
                    if(cost<cost_range and latter_data[col,1]>former_positions[row,1]):
                        temp=np.concatenate((former_positions[row], latter_data[col], [0]))
                        ex_matched_pair.append(temp)
                        temp=np.array([temp])
                        vector = self.calculate_vector(temp, 0)   
                        self.creat_new_record(latter_data[col:col+1], latter_elps_ax[col:col+1], vector)
                        self.remove_old_record(former_positions[row:row+1])
                    elif latter_data[col,1]>former_positions[row,1]:
                        costs = calculate_cost (prediction[row:row+1],latter_data[col:col+1],former_positions[row:row+1],dis_w=0.5,deg_w=0.5,area_w=0)
                        if(costs[0][0])<15:
                            temp=np.concatenate((former_positions[row], latter_data[col], [0]))
                            ex_matched_pair.append(temp)
                            temp=np.array([temp])
                            vector = self.calculate_vector(temp, 0)   
                            self.creat_new_record(latter_data[col:col+1], latter_elps_ax[col:col+1], vector)
                            self.remove_old_record(former_positions[row:row+1])
                    else:
                        self.costdata.append([compose_img_name,former_positions[row,0],former_positions[row,1],former_positions[row,2],
                                                    latter_data[col,0],latter_data[col,1],latter_data[col,2],
                                                    cost,cost_range,prediction[row,0],prediction[row,1]])
                        self.remove_old_record(former_positions[row:row+1])
                ex_matched_pair= np.array(ex_matched_pair).reshape(-1, 7)
                #經過校正後的結果刪除掉ann配對結果中相同的欄位避免出現不同配對結果
                mask = np.zeros(len(matched_pair), dtype=bool)
                for row in ex_matched_pair:
                    mask |= np.any([
                        np.all(matched_pair[:, :3] == row[:3], axis=1),  #前三個
                        np.all(matched_pair[:, 3:6] == row[3:6], axis=1) #後三個
                    ], axis=0)
                # 刪除匹配的行
                matched_pair= matched_pair[~mask]
            else:
                ex_matched_pair=np.array([]).reshape(0, 7)
            ex_matched_pair[:, -1] = self.calculate_velocity(ex_matched_pair, default_ = True)
            #紀錄新雨滴 
                                     
            for j in range(matched_pair.shape[0]):        # matched_pair : fx, fy, farea, lx, ly, larea, velocity
                vector = self.calculate_vector(matched_pair, j)
                ax = self.find_elps_axis(matched_pair[j, 3 : -1], latter_data, latter_elps_ax)#藉由配對到的雨滴與資料找回長短軸資訊
                self.creat_new_record(matched_pair[j, 3 : -1], ax, vector)                
            
            matched_pair = self.process_matched_pair(matched_pair, ex_matched_pair, NUM_FEATURES)
            self.refresh_memory()
            return matched_pair
        else:
            
            self.matched_memory[compose_img_name] = np.array([]).reshape(0, 7)  # 以compose_img_name為索引，存空的array，表示無配對組合
            self.compose_img_name_ = compose_img_name                           # compose_img_name_ : 將這次的compose_img_name暫存起來
                                                                                # 下一次呼叫到這個變數時意義為 : 上一次連續兩偵組合的檔名
            self.reset_memory()
            return np.array([]).reshape(0, NUM_FEATURES)
  
    def calculate_diameter(self, target):
        return (8.13e-2)*(((target[:, 4]**2)*target[:, 3])**(1 / 3))     # Deq = (6V/pi)^(1/3) = (l^2 * s)^(1/3)   # from papper : HSIV System for Rainfall Measurement

    def calculate_vector(self, matched_pair, i):  # matched_pair 2-d array
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
        new_target = copy.deepcopy(target)           # 2-d array so it need deep copy
        new_target = new_target.reshape(1, -1)
        if ax.shape[1] > 2: ax = ax[0, : 2]          # ax:長短軸  #防呆
        ax = ax.reshape(1, -1)
        key = self.trans(new_target)                 # new_target轉成 list再轉乘 str ，做為memory的key
        if key not in self.memory: self.memory[key] = (vector, new_target, ax)
        if key not in self.new_memory: self.new_memory[key] = (vector, new_target, ax)

    def find_elps_axis(self, target, latter_data, latter_elps_ax): # 3 parameter:2d array
        target = target.reshape(1, -1)
        key = (latter_data == target)[:, : 2]                      # bool
        return latter_elps_ax[key].reshape(1, -1)

    def find_time(self, time, mode):                               # 改效能這邊要改一下
        output = []
        if mode == 'MIN':
            for key in list(self.full_memory.keys()):
                if key[11 : -12] == time: output.append(self.full_memory[key])
        elif mode == 'HR':
            for key in list(self.rainfall_minute.keys()):
                if key[: -2] == time: output.append(self.rainfall_minute[key])
        return output

    def process_matched_pair(self, matched_pair, ex_matched_pair, NUM_FEATURES):
        matched_pair[:, -1] = self.calculate_velocity(matched_pair, default_ = True)
        matched_pair = np.vstack((matched_pair, ex_matched_pair))       # @@? 甚麼情況存在ex_matched_pair包含了matched_pair沒有的配對
        if matched_pair.shape[0] != 0: matched_pair = np.unique(matched_pair, axis = 0)
        return self.remove_zero(matched_pair, mode = 'complete')

    def record_rainfall_in_minute(self, compose_img_name):
        minute = compose_img_name[11 : -14]
        data = self.counted_raindrop[compose_img_name] #data:[[前一顆中心點x,前一顆中心點y,面積,下一顆中心點x,下一顆中心點y                                                      #長短軸,下一顆]]
        if data.shape[0] != 0:                         #        ,速度,直徑]
            short_axis = (6.486e-2)*data[:, 3]
            long_axis = (6.486e-2)*data[:, 4]
            tmp_rainfall = (np.pi / 6)*np.sum((long_axis**2)*short_axis)    # @@?
            tmp_rainfall = (tmp_rainfall*30) / (190*103.8)  #130*39
            tmp_num_raindrop = data.shape[0]*30                             # @@?
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


    def remove_bugs(self, target):
        target_col = target.shape[1]
        nonbug = target[:, -1] >= 0.15   # 粒徑要大於0.15
        nonbug = np.matlib.repmat(nonbug.reshape(-1, 1), 1, target.shape[1])
        return target[nonbug].reshape(-1, target_col)

    def remove_zero(self, target, mode, need_sub = 0):
        col = target.shape[1]      # 75*7 雨滴配對max
        nonzeros = (target != 0)   # target 不為0的index值為ture
        if mode == 'complete':
            nonzeros = np.sum(nonzeros, axis = 1) == col
        elif mode == 'uncomplete':
            nonzeros = np.sum(nonzeros, axis = 1) == (col - need_sub) # 每行相加(注意: 每行相加代表橫向的值全部相加)
                                                                      # uncomplete 時每列只會有前(col - need_sub)行有值，所以相加時要以 (col - need_sub) 為條件 
        nonzeros = np.matlib.repmat(nonzeros.reshape(-1, 1), 1, col)  # 布林值的mask
        return target[nonzeros].reshape(-1, col)                      # 保留ture的那一列

    def remove_old_record(self, target):
        del self.memory[self.trans(target)]

    def refresh_memory(self):
        self.memory = copy.deepcopy(self.new_memory)  # 深複製 (deep copy) 建立一份完全獨立的變數，資料型態不會被改變
        self.new_memory = {}

    def reset_memory(self):
        self.memory = {}
        self.new_memory = {}

    def trans(self, target):
        return str(target.tolist())

def save_run_info(read_path, start_time, finish_time, spend_time, threshold, save_arr_img, rainfall_area, is_modify, data_num, file_path=None):
    file_path = file_path + "run_info.txt"
    if file_path:
        smode = 'a'
        output = open(file_path, smode)
    else:
        output = sys.stdout
    header1 = "\n=========================================================================================================\n\n"
    row_template_0 = "{read_path}\n"
    row_template_1 = "start time            :  {start_time}\n"
    row_template_2 = "total # of image      :  {total_num}\n"
    row_template_3 = "processing_end_time   :  {finish_time}\n"
    row_template_4 = "spand_time            :  {spend_time_hr} hr : {spend_time_min} min : {spend_time_sec} sec\n"
    row_template_5 = "threshold_val         :  {threshold}\n"
    row_template_6 = "Cross-sectional area  :  {rainfall_area}\n"
    row_template_7 = "Modify                :  {is_modify}\n"
    row_template_8 = "SAVE 連線圖            :  {save_arr_img}\n"
    row_template_9 = "處理速度               :  {rate} 張/ 秒\n"

    output.write(header1)
    output.write(row_template_0.format(read_path=(read_path)))
    output.write(row_template_1.format(start_time=(time.ctime(start_time))))
    output.write(row_template_2.format(total_num=data_num))
    output.write(row_template_3.format(finish_time=(time.ctime(finish_time))))
    output.write(row_template_4.format(spend_time_hr=(int(spend_time // 3600)),spend_time_min=(int((spend_time % 3600) // 60)),spend_time_sec=(int(spend_time % 60)) ))
    output.write(row_template_5.format(threshold=(threshold)))
    output.write(row_template_6.format(rainfall_area=(rainfall_area)))
    output.write(row_template_7.format(is_modify=str(is_modify)))
    output.write(row_template_8.format(save_arr_img=bool(save_arr_img)))
    output.write(row_template_9.format(rate=(data_num/spend_time)))
    output.write(header1)
    if file_path:
        output.close()


    
def modify_by_velocity(l,s,velocity):
    out_l = 0
    out_s = 0
    deltapc = 0
    new_long = 0
    deltapc=velocity*1000*15.4*(69e-6)  #12.3->15.4
    new_long=l-deltapc
    if (new_long < 0):
        out_l = l
        out_s = s
    elif (new_long< s) :
        out_l = s
        out_s = new_long
    else :
        out_l = new_long
        out_s = s
    return out_l,out_s

def HSIV_QC(target_vel, target_dia):
       nonbug_dia = target_dia >= 0.15   # 粒徑要大於0.15
       #QV=-0.1021+4.932*target[:, -1]-0.9551*(target[:, -1]**2)+0.07934*(target[:, -1]**3)-0.002362*(target[:, -1]**4)
       QV=9.65-10.3*np.exp(-0.6*target_dia.astype(float))
       PQV=QV*(1+0.5)
       NQV=QV*(1-0.5)
       nonbug = PQV>= target_vel >= NQV
       nonbug = nonbug*nonbug_dia
       return nonbug.astype(bool)

def get_rain_data(trackedmem, shadow_fix_en, qc_en):
    raindata={}
    for value in trackedmem.values():
        cx,cy,area=value.boundingbox[0][0],value.boundingbox[0][1],value.boundingbox[0][2]
        vec_x,vec_y=value.vector[0][0],value.vector[0][1]
        velocity=(6.486e-5)*np.sqrt((vec_x)**2 + (vec_y)**2)*500
        angle, maxdelta_angle = value.angle, value.maxdelta_angle
        s,l=value.ax[0][0],value.ax[0][1]
        if shadow_fix_en : 
            l,s=modify_by_velocity(l,s,velocity)
        diameter=(6.486e-2)*(((l**2)*s)**(1 / 3))
        name=value.compose_img_name
        name=int(name[11:17]+name[18:21])
        l=[value.compose_img_name,cx,cy,area,s,l,vec_x,vec_y,velocity,diameter,angle, maxdelta_angle]
        if qc_en: 
            QC_Data_vld = HSIV_QC(velocity, diameter)
            if (name not in raindata.keys()) and QC_Data_vld:
                raindata[name]=[l]
            elif QC_Data_vld :
                raindata[name].append(l)
        else: 
            if (name not in raindata.keys()):
                raindata[name]=[l]
            else:
                raindata[name].append(l)
    raindata=dict(sorted(raindata.items()))
    return raindata
def record_raindrop_second(raindata, csvfile):
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(csvfile),'a',newline='',encoding="utf-8") as f1:
            writer=csv.writer(f1)
            for value in raindata.values():
                for row in value:
                    writer.writerow(row)
def record_raindrop_min(raindata, csvfile):
    min=0
    min_number=0
    min_rainfall=0
    rainfall=0
    number=0
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(csvfile),'a',newline='',encoding="utf-8") as f2:
        writer=csv.writer(f2)             
        for values in raindata.values():  
            for value in values:    #value: [compose_img_name,cx,cy,area,s,l,vec_x,vec_y,velocity,diameter]
                if min>0:
                    current_min=int(value[0][13:15])
                    if(current_min-min!=0):
                        min_number=0
                        min_rainfall=0    
                min=int(value[0][13:15])
                hourmin=value[0][11:15]
                short_axis = (6.486e-2)*value[4]
                long_axis = (6.486e-2)*value[5]
                tmp_rainfall = (np.pi / 6)*np.sum((long_axis**2)*short_axis)     
                tmp_rainfall = (tmp_rainfall*30) / (190*103.8)#
                rainfall+=tmp_rainfall
                min_rainfall+=tmp_rainfall
                min_number+=30
                number+=30
                writer.writerow([hourmin,min_rainfall,min_number,rainfall,number])
def record_raindrop_hour(csvfilemin,csvfilehour):
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(csvfilemin),'r',newline='',encoding="utf-8") as f2:
        raindata=[]
        reader = csv.reader(f2)
        write=False
        last_hour=0
        Cumulative_rainfall=0
        number=0
        for col,row in enumerate(reader):
            if(last_hour>0):
                current_hour=int(row[0][:2])
                if(current_hour-last_hour>=1):
                    if(len(raindata)==0):
                        raindata.append([last_hour,Cumulative_rainfall,number,Cumulative_rainfall,number])
                        write=True
                    else:
                        total_rainfall=Cumulative_rainfall-raindata[-1][3]
                        total_number=number-raindata[-1][4]
                        raindata.append([last_hour,total_rainfall, total_number,Cumulative_rainfall,number])
                        write=True
            if(col >0):
                last_hour=int(row[0][:2])
                Cumulative_rainfall=float(row[3])
                number=float(row[4])
        if(write==True):
            total_rainfall=Cumulative_rainfall-raindata[-1][3]
            total_number=number-raindata[-1][4]
            raindata.append([last_hour,total_rainfall, total_number,Cumulative_rainfall,number])
        elif(write==False):
            raindata.append([last_hour,Cumulative_rainfall,number,Cumulative_rainfall,number])
    with open( WRITE_PATH_ANALYZE + '{}.csv'.format(csvfilehour),'a',newline='',encoding="utf-8") as f3:
                writer=csv.writer(f3)
                writer.writerows(raindata)

def get_rainfall_ARR (result_id, write_path, write_path_analyze,write_path_arrimg ):
    radius = 10
    if not os.path.exists(write_path + "Raninfall_ARR" + result_id + "/"): os.makedirs(write_path + "Raninfall_ARR"+result_id+"/")
    sec_raindrops_df = pd.read_csv(write_path_analyze+ "雨滴每秒結果"+result_id+".csv")
    sec_raindrops_df = sec_raindrops_df.to_dict('records')
    # data
    rainfall_info_dict={}
    data_dict = {}
    for item in sec_raindrops_df:
        i=0
        new_key = item['檔案名稱']
        l=[]
        for key , values in item.items():
            i += 1
            if (i == 2 ) : x=values
            if (i == 3 ) : y=values
        l = (x,y)
        if new_key not in rainfall_info_dict.keys():
            rainfall_info_dict[new_key]=[l]
        else:
            rainfall_info_dict[new_key].append(l)
    color=(255, 192, 203)     
    for key, values in rainfall_info_dict.items():
        t_name      = key[0:18]
        former_name = key[18:21]
        latter_name = key[22:25]
        int_f_name =  int(key[18:21])
        if not(latter_name == '000'):
            source_path = write_path_arrimg + "Arrow_" + key
            if os.path.exists(source_path):
                target_path = write_path + "Raninfall_ARR"+result_id+"/" + "Arrow_" + key
                shutil.copy(source_path, target_path)
        elif (latter_name == '000'):
            latter_name = key[18:21]
            int_f_name =  int(key[18:21])-1
            former_name = str(int_f_name).zfill(3)
            compose_name = t_name+former_name+"_"+latter_name+".png"
            source_path = write_path_arrimg + "Arrow_" + compose_name
            if os.path.exists(source_path):
                target_path = write_path + "Raninfall_ARR" + result_id + "/" + "Arrow_" + compose_name
            #shutil.copy(source_path, target_path)
        if os.path.exists(source_path):
            image = Image.open(source_path)
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("arial.ttf",30)
            except IOError:
                font = ImageFont.load_default()
            for (x, y) in values:
                tx = x* 3.85 + 407;     ty = y* 3.85 + 287
                int(tx)
                int(ty)
                draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=color, outline=color)
                draw.text((tx + 30, ty - 30), f"({x}, {y})", fill='yellow', font=font )
            image.save(target_path)


def compose_ARR_img(result_id, write_path, write_path_arrimg):        # compose_ARR_img(RESULT_ID, RESULT_PATH, RESULT_PATH_ARRIMG)
  if not os.path.exists(write_path + "Track_ARR" + result_id + "/"): os.makedirs(write_path + "Track_ARR" + result_id + "/")
  is_continue = 0
  pre_is_continue = 0
  count_limit = 2
  curr_img = np.zeros((640, 480, 3), dtype = np.uint8)
  next_img = np.zeros((640, 480, 3), dtype = np.uint8)
  compose_img = np.zeros((640, 480, 3), dtype = np.uint8)

  start = time.time()
  print("\n",write_path_arrimg,"\n")
  file_list = sorted(os.listdir(path = write_path_arrimg))
  del_ds_store(file_list)
  file_list_sizes = len(file_list)
  img_count = 0

  for idx in range (0,int(file_list_sizes),1):
    is_continue = 0
    if idx <= int(file_list_sizes) -2 :
      curr_img_name = file_list[idx]
      next_img_name = file_list[idx+1]
      curr_img_name_l = curr_img_name[-7:  -4]
      next_img_name_f = next_img_name[-11: -8]
      if (curr_img_name[: -12] == next_img_name[: -12])  and (curr_img_name_l == next_img_name_f) and img_count < (count_limit-1):       # 影像連續
        is_continue = 1
        img_count += 1
        if is_continue == 1 and pre_is_continue == 0 :
          first_img_name = curr_img_name[:-7]
          curr_img = cv2.imread(write_path_arrimg + curr_img_name)
          next_img = cv2.imread(write_path_arrimg + next_img_name)
          compose_img = curr_img + next_img
        elif is_continue == 1 and pre_is_continue == 1 :
          next_img = cv2.imread(write_path_arrimg + next_img_name)
          compose_img = compose_img + next_img
      elif is_continue == 0 and pre_is_continue == 0:
        curr_img = cv2.imread(write_path_arrimg + curr_img_name)
        cv2.imwrite(write_path + "Track_ARR" + result_id + "/" + curr_img_name, curr_img)
        img_count = 0
      else:
        is_continue = 0
        cv2.imwrite(write_path + "Track_ARR" + result_id + "/" + first_img_name + curr_img_name_l + ".png", compose_img)
        img_count = 0
      pre_is_continue = is_continue
    
    else:
      if pre_is_continue == 1 :
        curr_img_name = file_list[idx]
        curr_img_name_l = curr_img_name[-7:  -4]
        cv2.imwrite(write_path + "Track_ARR" + result_id + "/" + first_img_name + curr_img_name_l  + ".png", compose_img)
        img_count = 0
      else :
        curr_img_name = file_list[idx]
        curr_img = cv2.imread(write_path_arrimg + curr_img_name)
        cv2.imwrite(write_path + "Track_ARR" + result_id + "/" + curr_img_name, curr_img)
        img_count = 0

def get_rainfall_ARR_track (result_id, write_path, write_path_analyze,write_path_arrimg ):
  radius = 10
  color  =(255, 192, 203)
  rainfall_info_dict = {}
  sec_raindrops_df   = pd.read_csv(write_path_analyze+ "雨滴每秒結果"+result_id+".csv")
  sec_raindrops_df   = sec_raindrops_df.to_dict('records')
  rainfall_info_dict = load_swc_data(sec_raindrops_df)

  file_list       = sorted(os.listdir(path = write_path + "Track_ARR" + result_id + "/"))
  del_ds_store(file_list)
  file_list_sizes = len(file_list)
  idx             = 0

  compose_image = cv2.imread(write_path + "Track_ARR"+ result_id + "/" + file_list[0])
  for key, values in rainfall_info_dict.items():
      temp_name   = key[0:18]
      former_name = key[18:21]
      latter_name = key[22:25]
      int_f_name  = int(key[18:21])

      compose_name = file_list[idx]
      compose_name_f = compose_name[-11: -8]
      compose_name_l = compose_name[-7:  -4]

      if (int(compose_name_f) <= int_f_name <= int(compose_name_l) ) and compose_name[6:-12] == key[:17] :
        pass
      else:
        cv2.imwrite(write_path + "Track_ARR"+ result_id + "/"+ compose_name, compose_image)
        while not ((int(compose_name[-11: -8]) <= int_f_name <= int(compose_name[-7:  -4]) ) and (compose_name[6:-12] == key[:17])):
          idx = idx + 1
          compose_name = file_list[idx]
        compose_image = cv2.imread(write_path + "Track_ARR"+ result_id + "/" + compose_name)

      # opencv 版本
      for (x, y) in values:
      # 標記圓點
        tx = x* 3.85 + 407;     ty = y* 3.85 + 287
        compose_image = cv2.circle(compose_image, (int(tx), int(ty)), radius, (255, 192, 203) , -1)
        # 添加文字標註
        text = f"({x},{y})"
        text_position = (int(tx) + 20, int(ty) - 20)  # 調整文字位置
        compose_image = cv2.putText(compose_image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 225), 1, cv2.LINE_AA)

      if(idx == len(rainfall_info_dict)):
         cv2.imwrite(write_path + "Track_ARR"+ result_id + "/"+ compose_name, compose_image)

def load_swc_data(dict_in):
  dict_out={}
  data_dict = {}
  for item in dict_in:
    i=0
    new_key = item['檔案名稱']
    l=[]
    for key , values in item.items():
           i += 1
           if (i == 2 ) : x=values
           if (i == 3 ) : y=values
    l = (x,y)
    if new_key not in dict_out.keys():
       dict_out[new_key]=[l]
    else:
       dict_out[new_key].append(l)
  return dict_out

def coor_filter(input_data):
    all_len = input_data.shape[0]
    coor_filter_binary = np.zeros((all_len,1), dtype=bool)
    for i in range (all_len):
        coor_filter_binary[i] = True if (input_data[i, 1] < input_data[i, 4])   else False
    return coor_filter_binary.reshape(-1,1)
def check_match(WRITE_PATH,WRITE_PATH_ANALYZE,WRITE_PATH_ARRIMG):
    mismatch=WRITE_PATH+"mismatch"
    if not os.path.exists(mismatch):
        os.makedirs(WRITE_PATH+"mismatch")
    filename=[]
    check={}
    with open(WRITE_PATH_ANALYZE+"cost.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if(i>0):
                filename.append(["Arrow_"+row[0],float(row[1]),float(row[2]),float(row[3]),
                                 float(row[4]),float(row[5]),float(row[6]),
                                 round(float(row[7]), 2),round(float(row[8]), 2),float(row[9]),float(row[10])])
    for i, data in enumerate(filename):
        source_path=WRITE_PATH_ARRIMG+data[0]
        destination_path=mismatch+'/'+data[0]
        if(source_path in check):
            destination_path=source_path[:-4]+str(check[source_path])+'.png'
            check[source_path]+=1
        if os.path.exists(source_path):
            image=cv2.imread(source_path)
            desimage=image.copy()
            tx1 = data[1]* 3.85 + 407;     ty1 = data[2]* 3.85 + 287
            tx2 = data[4]* 3.85 + 407;     ty2 = data[5]* 3.85 + 287
            text_position1 = (int(tx1)+20, int(ty1)+40)
            text_position2 = (int(tx1)+20, int(ty1)+80)  
            text_position3 = (int(tx1)+20, int(ty1)+120)
            text_position4=(int(tx2)+10, int(ty2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            text_color = (255, 255, 255)  
            thickness = 3
            text1="former=("+str(int(data[1]))+','+str(int(data[2]))+','+str(int(data[3]))+")"
            text2="prediction=("+str(int(data[9]))+','+str(int(data[10]))+")"
            text3="COST="+str(data[7])
            text4="latter=("+str(int(data[4]))+','+str(int(data[5]))+','+str(int(data[6]))+")"
            cv2.putText(desimage, text1, text_position1, font, font_scale, text_color, thickness)
            cv2.putText(desimage, text2, text_position2, font, font_scale, text_color, thickness)
            cv2.putText(desimage, text3, text_position3, font, font_scale, text_color, thickness)
            cv2.putText(desimage, text4, text_position4, font, font_scale, text_color, thickness)
            cv2.imwrite(destination_path, desimage)
            check[source_path]=1