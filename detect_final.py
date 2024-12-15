# // +FHDR--------------------------------------------------------------------------------------------------------- //
# // Project ____________                                                                                           //
# // File name __________ detect_final.py                                                                           //
# // Creator ____________ Yan, Wei-Ting                                                                             //
# // Built Date _________ 01-07-2024                                                                                //
# // Function ___________ rename_files_add_zero                                                                     //
# //                      gen_data_image                                                                            //
# //                      verify_ARR_img_result                                                                     //
# //                      verify_detected_Data_img                                                                  //
# //                      static_measurement_bounding_pair                                                          //
# //                                                                                                                //
# // Hierarchy __________                                                                                           //
# //   Parent ___________                                                                                           //
# //   Children _________                                                                                           //
# // Revision history ___ Date        Author            Description                                                 //
# //                  ___                                                                                           //
# // -FHDR--------------------------------------------------------------------------------------------------------- //
# //+...........+...................+.............................................................................. //
# //4...........16..................36............................................................................. //
import os
import sys
import copy
import time
import getpass
import gc
import numpy as np
import pandas as pd
import cv2
from skimage import morphology
from skimage.feature import canny
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from sklearn.metrics import confusion_matrix
from path_def import detect_raindrop_path_def

READ_DATA_PATH, WRITE_R_PATH, WRITE_D_PATH, WRITE_THERSHOLD_PATH = detect_raindrop_path_def()

# >>>>>>>>>>>>>>>>>>>> var >>>>>>>>>>>>>>>>>>>> #
diffThreshold      = int(5)
maxval             = int(255)
areaThreshold      = int(15)
framerate          = int(500)
saveOriginalImg    = framerate
saveProcessedImg   = int(500)
# <<<<<<<<<<<<<<<<<<<< var <<<<<<<<<<<<<<<<<<<< #

closing_kernel = morphology.disk(radius = 7)

# ***********************/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**
#  function             /**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/***
# *********************/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/****
def rename_files_add_zero(path):
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            parts = filename.split('_')
            last_part = parts[-1].split('.')[0]
            new_last_part = last_part.zfill(3)
            new_filename = '_'.join(parts[:-1]) + '_' + new_last_part + '.png'
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

def check_path_exist(check_path):
    if not os.path.exists(check_path):
        os.makedirs(check_path)

def del_ds_store(file_list):
    if file_list[0] == '.DS_Store': file_list.remove('.DS_Store')


def gen_data_image( readpath, write_r_path, write_d_path, write_threshold_path):
  idx = int(0)
  sum = int(0)
  j   = int(0)
  iter = 500
  
  before_median = np.zeros([480, 640])
  
  start = time.time()
  print("\n",readpath,"\n")
  file_list = sorted(os.listdir(path = readpath))
  del_ds_store(file_list)
  file_list_sizes = len(file_list)
  
  pre_image = cv2.imread(readpath+file_list[0])
    
  for idx in range (0,int(file_list_sizes),1):  
      start_iter =time.time()
      
      now_image_name = file_list[idx]
      now_image = cv2.imread(readpath + now_image_name)

      if(now_image_name[18:21] == "001"):
         pre_image = now_image.copy()

      gray_pre_image = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
      gray_now_image = cv2.cvtColor(now_image, cv2.COLOR_RGB2GRAY)

      pre_image = now_image.copy()

      before_median = cv2.subtract(gray_now_image , gray_pre_image)
      
      imDiff = before_median.copy()
      median_MINUS = cv2.medianBlur(before_median, 5)

      median_final        = median_MINUS.copy()
      median_to_th_binary = median_MINUS.copy()
      
      ret1,threshold_img       = cv2.threshold(median_final, diffThreshold,maxval, cv2.THRESH_TOZERO)
      ret2,median_to_th_binary = cv2.threshold(median_to_th_binary, diffThreshold, 255,cv2.THRESH_BINARY)

      
      withborder = threshold_img.copy()
      contours,hierarchy                = cv2.findContours(image=threshold_img , mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)    # opencv 4.x
      #threshold_img, contours,hierarchy = cv2.findContours(image=threshold_img , mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)    # opencv 3.4.x
      
      cv2.drawContours(image=withborder, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_8)
      
      sum  = np.sum(withborder)
      sum2 = np.sum(median_to_th_binary)
      
      if (sum2 > areaThreshold ) :
        save_name_t = "T" + now_image_name[1:21] + ".png"        
        cv2.imwrite(write_threshold_path + save_name_t, median_to_th_binary)
      
      if (sum > areaThreshold ) :
        save_name_r = "R" + now_image_name[1:21] + ".png"
        save_name_d = "D" + now_image_name[1:21] + ".png"
  
        cv2.imwrite(write_r_path + save_name_r, withborder)
        cv2.imwrite(write_d_path + save_name_d, imDiff)

      if ( (idx%iter)==0):
        end_iter = time.time()
        print(f"Iteration: {j}\tTime taken: {(end_iter-start_iter)*10**3:.03f}ms")
        j = j + 1
      
      if ( (idx%(5000))==0):
        del hierarchy
        del contours
        del withborder
        del ret1
        del ret2
        gc.collect()
        print(f"clean memory")

  end = time.time()
  print("The time of execution of above program is :",(end-start) * 10**3, "ms")
  print("--------------")

# ***********************/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**
#  pre work             /**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/***
# *********************/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/****    
check_path_exist(WRITE_R_PATH)
check_path_exist(WRITE_D_PATH)
check_path_exist(WRITE_THERSHOLD_PATH)
rename_files_add_zero(READ_DATA_PATH)

# ***********************/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**\**\****/**/**
#    Main               /**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/****\**\**/**/***
# *********************/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/******\**\/**/****
gen_data_image(READ_DATA_PATH, WRITE_R_PATH, WRITE_D_PATH, WRITE_THERSHOLD_PATH)



