# """
# Created on 2024/05/11 

# @author: matthew
# """

import sys
import getpass
import pandas as pd
import numpy as np
from QC_utils import sum_hr_info, update_SV_o,init_SV_gif,add_tital,write_CSV,plot_SV_joint_distribution
from QC_utils import plot_SV_joint_distribution
from QC_utils import *
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import matplotlib
# load .mat file
from scipy.io import loadmat


# ===========================================
# user conf.
TIME_SEL   = "18:09"
MIN_OUT_EN = 0
CHANGE_EN  = 1
PATH_ROOT = "F:/Raindrop_folder/Rainfall_project_2023/Parsivel_RAW_data/"
TARG_PATH = "F:/Raindrop_folder/Rainfall_project_2023/Parsivel_QC_data/"
FILE_NAME = "Parsivel_20240501"

# ============================================================================================================================================================================
new_header = ["TIMESTAMP", "rIntensity", "nParticles"] + [f"drop({i//32+1},{i%32+1})" for i in range(1024)]
aD=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5 ,7.5 ,8.5 ,9.5 ,11,13,15,17,19,21,24.5])
dD=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,3,3,3,])
aV=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.10,1.30,1.50,1.70,1.90,2.20,2.60,3,3.40,3.80,4.40,5.20,6,6.80,7.60,8.80,10.4,12,13.6,15.2,17.6,20.8,])
dV=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,])
sTIME      = 0
rINTENSITY = 1
nPARTICLES = 2

def from_NTU_data():
    file_paths = ['Parsivel_20240501']            #未來用選擇方式匯入
    if(len(file_paths)>1):
      new_file_name = file_paths[0][:]+"-"+file_paths[-1][6:]
    else:  
      new_file_name = file_paths[0][:]
    # Create a list to store the data from each file
    data_frames = []

    # Iterate over each file path, read the data, and append to the list
    for path in file_paths:
      # Read the file
      df = pd.read_csv(PATH_ROOT+path+".csv", header=None)
      # Ensure header consistency across all data frames by using the new header
      df.columns = new_header[:df.shape[1]]  # Using the header from the previous step
      data_frames.append(df)

    # Concatenate all data frames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_csv_path = PATH_ROOT+new_file_name+'_RAW.csv'
    combined_df.to_csv(combined_csv_path, index=False)
    return combined_df, new_file_name

def change_data_format (data_frame_in, file_name_out):
  #RAW_df = pd.read_csv(PATH_ROOT+ file_name_out +'_RAW.csv')
  data_dict_copy = pd.DataFrame.from_dict(data_frame_in) 
  #RAW_data_dict = RAW_df.set_index('TIMESTAMP').T.to_dict('list')
  RAW_data_dict = data_frame_in.set_index('TIMESTAMP').T.to_dict('list')
  data_dict = RAW_data_dict.copy() 
  # ------------------------------------------------------------------------------
  #Date_str = FILE_NAME[9:13] + '/' + FILE_NAME[13:15] + '/' + FILE_NAME[15:17]
  date_format = "%Y/%m/%d"
  for key, values in data_dict.items():
    nPTC = 0                                  #  每列顆粒數
    nPTC = sum(values[2:])
    values[1] = nPTC
    if key[11:13] == '24':
      part_date = key[0:10]
      date_obj = datetime.strptime(part_date, date_format) + timedelta(days=1)
      date_obj = date_obj.strftime(date_format)
      new_key = date_obj + " 00" + key[13:]
      data_dict.update({new_key:data_dict.pop(key)})
  update_dict = {key: data_dict[key] for key in sorted(data_dict, key=lambda date: datetime.strptime(date, '%Y/%m/%d %H:%M:%S'))}
  data_dict = pd.DataFrame.from_dict(update_dict, orient='index', columns=data_dict_copy.columns[1:]) # 將DataFrame轉換為字典
  data_file_path = PATH_ROOT + file_name_out + "_mRAW.csv"
  data_dict.to_csv(data_file_path,index_label='TIMESTAMP')   # data_dict 要是DataFrame 才可以用to_csv


new_combined_df, comb_name = from_NTU_data()
change_data_format(new_combined_df,comb_name)    









































''' 
def change_data_format(data_frame_in, file_name_out):
    RAW_df = pd.read_csv(PATH_ROOT + FILE_NAME + '_RAW.csv')
    # 保留 TIMESTAMP 為一個列，不將其設為索引
    #RAW_data_dict = RAW_df.to_dict('records')
    RAW_data_dict = data_frame_in.to_dict('records')
    
    # 新的 data_dict 使用普通字典而不是以 TIMESTAMP 為索引的字典
    data_dict = {}
    date_format = "%Y/%m/%d"
    for item in RAW_data_dict:
        timestamp = item['TIMESTAMP']
        nPTC = 0
        i    = 0
        for key , values in item.items():
            i += 1
            if (i >= 4) : nPTC += values
        # 進行時間的轉換和更新
        if timestamp[11:13] == '24':
            date_obj = datetime.strptime(timestamp[:10], date_format) + timedelta(days=1)
            new_timestamp = date_obj.strftime(date_format) + " 00" + timestamp[13:]
            item['TIMESTAMP'] = new_timestamp
        item['nParticles']   = nPTC
        data_dict[timestamp] = item

    # 將 data_dict 轉換回 DataFrame
    new_RAW_df = pd.DataFrame.from_dict(list(data_dict.values()))
    # 重新排序，這裡假設 data_dict 的值是一個包含所有原始列數據的列表
    new_RAW_df.sort_values(by='TIMESTAMP', inplace=True)
    
    # 將更新後的 DataFrame 寫入 CSV，保留所有列
    data_file_path = PATH_ROOT + file_name_out + "_mRAW.csv"
    new_RAW_df.to_csv(data_file_path, index=False)
'''