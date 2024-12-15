import csv
import pandas as pd
import numpy as np

FILE_NAME     = "Parsivel_20240331-17TH"
VELOCITY_MASK = 1


def change_data_format ():
  # 第一個檔案的路徑
  file1_path = 'D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/'+FILE_NAME+'.csv'

  # 定義需要保留的欄位名稱
  columns_to_keep = [ 'TIMESTAMP', 'sTime', 'rIntensity', 'nParticles']  \
                    + [f'drop({i},{j})' for i in range(1, 33) for j in range(1, 33)]

  # 載入並調整第一個檔案，移除不需要的行並過濾欄位
  file1_df_reloaded    = pd.read_csv(file1_path, skiprows=[0, 2, 3])
  filtered_columns     = [col for col in file1_df_reloaded.columns if col in columns_to_keep]
  file1_adjusted_final = file1_df_reloaded[filtered_columns]

  # 保存調整後的檔案
  adjusted_file_path = 'D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/'+ FILE_NAME +'_Nview.csv'
  file1_adjusted_final.to_csv(adjusted_file_path, index=False)
  print(f"\nAdjusted file saved to: {adjusted_file_path}")

def Quality_ctrl ():
  adjusted_df = pd.read_csv('D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/'+ FILE_NAME +'_Nview.csv')

  # 將DataFrame轉換為字典
  data_dict = adjusted_df.set_index('TIMESTAMP').T.to_dict('list')

  # 顯示字典中的前幾個項目來確認結果
  temp_values = np.zeros(1027, dtype=int)
  new_DSD_fig = init_DSD_gif()

  # ======================================================== 刪除不合理雨滴大小 ========================================================
  for key, values in data_dict.items():
    temp_values = np.zeros(1027, dtype=int)
    for i in range (1024):
      y = int(i/32)
      x = int(i%32)
      if (x>=25 and x<=31):
        temp_values[i+3] = values[i+3]

    dia_totla_del = int(sum(temp_values))
    values[2] = int(values[2]) - dia_totla_del

  # ======================================================== 保留合理雨滴落速 ========================================================
  if (VELOCITY_MASK):
    for key, values in data_dict.items():
      temp_values = np.zeros(1027)
      for i in range(1024) :
        y = int(i/32)
        x = i%32
        if x == 1 and ( y >= 1 and y <= 7 ) :
          temp_values[i+3] = values[i+3]
        elif x == 2  and ( y >= 4 and y <= 15 ) :
          temp_values[i+3] = values[i+3]
        elif x == 3  and ( y >= 7 and y <= 17 ) :
          temp_values[i+3] = values[i+3]
        elif x == 4  and ( y >= 10 and y <= 19 ) :
          temp_values[i+3] = values[i+3]
        elif x == 5  and ( y >= 11 and y <= 20 ) :
          temp_values[i+3] = values[i+3]
        elif x == 6  and ( y >= 12 and y <= 21 ) :
          temp_values[i+3] = values[i+3]
        elif x == 7  and ( y >= 13 and y <= 22 ) :
          temp_values[i+3] = values[i+3]
        elif x == 8  and ( y >= 15 and y <= 23 ) :
          temp_values[i+3] = values[i+3]
        elif x == 9  and ( y >= 15 and y <= 24 ) :
          temp_values[i+3] = values[i+3]
        elif x == 10 and ( y >= 15 and y <= 25 ) :
          temp_values[i+3] = values[i+3]
        elif x == 11 and ( y >= 16 and y <= 25 ) :
          temp_values[i+3] = values[i+3]
        elif x == 12 and ( y >= 17 and y <= 26 ) :
          temp_values[i+3] = values[i+3]
        elif x == 13 and ( y >= 17 and y <= 26 ) :
          temp_values[i+3] = values[i+3]
        elif x == 14 and ( y >= 18 and y <= 27 ) :
          temp_values[i+3] = values[i+3]
        elif x == 15 and ( y >= 19 and y <= 27 ) :
          temp_values[i+3] = values[i+3]
        elif x == 16 and ( y >= 19 and y <= 28 ) :
          temp_values[i+3] = values[i+3]
        elif x == 17 and ( y >= 20 and y <= 28 ) :
          temp_values[i+3] = values[i+3]
        elif x == 18 and ( y >= 20 and y <= 28 ) :
          temp_values[i+3] = values[i+3]
        elif x == 19 and ( y >= 20 and y <= 28 ) :
          temp_values[i+3] = values[i+3]
        elif x == 20 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
        elif x == 21 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
        elif x == 22 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
        elif x == 23 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
        elif x == 24 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
        elif x == 25 and ( y >= 20 and y <= 29 ) :
          temp_values[i+3] = values[i+3]
      values[2:] = temp_values[2:]
      values[2] = np.sum(temp_values[3:])
  # ======================================================== 刪除非降雨事件 ========================================================

  for key, values in data_dict.items():
    # 分鐘顆數 (nParticles) 小於 10
    if (values[2]<10):
      for k in range (0,1026,1) :
         values[1+k] = 0
    # 分鐘降雨率 (rIntensity) 小於 0.1mm/h
    if(values[1] < 0.1):
      for k in range (0,1026,1) :
         values[1+k] = 0
    new_DSD_fig = update_DSD(new_DSD_fig, values)

  sum_data = np.zeros([32,4])
  for key, values in data_dict.items():
    hr = int(values[0][:2])
    if (hr == 0):
        sum_data[0][0] = 0
        sum_data[0][1] = sum_data[0][1] + values[1]
        sum_data[0][2] = sum_data[0][2] + values[2]
        sum_data[0][3] = sum_data[0][1] / 60
    elif (hr == 1):
        sum_data[1][0] = 1
        sum_data[1][1] = sum_data[1][1] + values[1]
        sum_data[1][2] = sum_data[1][2] + values[2]
        sum_data[1][3] = sum_data[1][1] / 60
    elif (hr == 2):
        sum_data[2][0] = 2
        sum_data[2][1] = sum_data[2][1] + values[1]
        sum_data[2][2] = sum_data[2][2] + values[2]
        sum_data[2][3] = sum_data[2][1] / 60
    elif (hr == 3):
        sum_data[3][0] = 3
        sum_data[3][1] = sum_data[3][1] + values[1]
        sum_data[3][2] = sum_data[3][2] + values[2]
        sum_data[3][3] = sum_data[3][1] / 60
    elif (hr == 4):
        sum_data[4][0] = 4
        sum_data[4][1] = sum_data[4][1] + values[1]
        sum_data[4][2] = sum_data[4][2] + values[2]
        sum_data[4][3] = sum_data[4][1] / 60
    elif (hr == 5):
        sum_data[5][0] = 5
        sum_data[5][1] = sum_data[5][1] + values[1]
        sum_data[5][2] = sum_data[5][2] + values[2]
        sum_data[5][3] = sum_data[5][1] / 60
    elif (hr == 6):
        sum_data[6][0] = 6
        sum_data[6][1] = sum_data[6][1] + values[1]
        sum_data[6][2] = sum_data[6][2] + values[2]
        sum_data[6][3] = sum_data[6][1] / 60
    elif (hr == 7):
        sum_data[7][0] = 7
        sum_data[7][1] = sum_data[7][1] + values[1]
        sum_data[7][2] = sum_data[7][2] + values[2]
        sum_data[7][3] = sum_data[7][1] / 60
    elif (hr == 8):
        sum_data[8][0] = 8
        sum_data[8][1] = sum_data[8][1] + values[1]
        sum_data[8][2] = sum_data[8][2] + values[2]
        sum_data[8][3] = sum_data[8][1] / 60
    elif (hr == 9):
        sum_data[9][0] = 9
        sum_data[9][1] = sum_data[9][1] + values[1]
        sum_data[9][2] = sum_data[9][2] + values[2]
        sum_data[9][3] = sum_data[9][1] / 60
    elif (hr == 10):
        sum_data[10][0] = 10
        sum_data[10][1] = sum_data[10][1] + values[1]
        sum_data[10][2] = sum_data[10][2] + values[2]
        sum_data[10][3] = sum_data[10][1] / 60
    elif (hr == 11):
        sum_data[11][0] = 11
        sum_data[11][1] = sum_data[11][1] + values[1]
        sum_data[11][2] = sum_data[11][2] + values[2]
        sum_data[11][3] = sum_data[11][1] / 60
    elif (hr == 12):
        sum_data[12][0] = 12
        sum_data[12][1] = sum_data[12][1] + values[1]
        sum_data[12][2] = sum_data[12][2] + values[2]
        sum_data[12][3] = sum_data[12][1] / 60
    elif (hr == 13):
        sum_data[13][0] = 13
        sum_data[13][1] = sum_data[13][1] + values[1]
        sum_data[13][2] = sum_data[13][2] + values[2]
        sum_data[13][3] = sum_data[13][1] / 60
    elif (hr == 14):
        sum_data[14][0] = 14
        sum_data[14][1] = sum_data[14][1] + values[1]
        sum_data[14][2] = sum_data[14][2] + values[2]
        sum_data[14][3] = sum_data[14][1] / 60
    elif (hr == 15):
        sum_data[15][0] = 15
        sum_data[15][1] = sum_data[15][1] + values[1]
        sum_data[15][2] = sum_data[15][2] + values[2]
        sum_data[15][3] = sum_data[15][1] / 60
    elif (hr == 16):
        sum_data[16][0] = 16
        sum_data[16][1] = sum_data[16][1] + values[1]
        sum_data[16][2] = sum_data[16][2] + values[2]
        sum_data[16][3] = sum_data[16][1] / 60
    elif (hr == 17):
        sum_data[17][0] = 17
        sum_data[17][1] = sum_data[17][1] + values[1]
        sum_data[17][2] = sum_data[17][2] + values[2]
        sum_data[17][3] = sum_data[17][1] / 60
    elif (hr == 18):
        sum_data[18][0] = 18
        sum_data[18][1] = sum_data[18][1] + values[1]
        sum_data[18][2] = sum_data[18][2] + values[2]
        sum_data[18][3] = sum_data[18][1] / 60
    elif (hr == 19):
        sum_data[19][0] = 19
        sum_data[19][1] = sum_data[19][1] + values[1]
        sum_data[19][2] = sum_data[19][2] + values[2]
        sum_data[19][3] = sum_data[19][1] / 60
    elif (hr == 20):
        sum_data[20][0] = 20
        sum_data[20][1] = sum_data[20][1] + values[1]
        sum_data[20][2] = sum_data[20][2] + values[2]
        sum_data[20][3] = sum_data[20][1] / 60
    elif (hr == 21):
        sum_data[21][0] = 21
        sum_data[21][1] = sum_data[21][1] + values[1]
        sum_data[21][2] = sum_data[21][2] + values[2]
        sum_data[21][3] = sum_data[21][1] / 60
    elif (hr == 22):
        sum_data[22][0] = 22
        sum_data[22][1] = sum_data[22][1] + values[1]
        sum_data[22][2] = sum_data[22][2] + values[2]
        sum_data[22][3] = sum_data[22][1] / 60
    elif (hr == 23):
        sum_data[23][0] = 23
        sum_data[23][1] = sum_data[23][1] + values[1]
        sum_data[23][2] = sum_data[23][2] + values[2]
        sum_data[23][3] = sum_data[23][1] / 60


  write_CSV("D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/"+FILE_NAME+"_SUM.csv",sum_data )


  new_DSD_fig = np.asanyarray(new_DSD_fig,float)
  new_DSD_fig = add_tital(new_DSD_fig)
  write_CSV ("D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/"+FILE_NAME+"_DSD_QC.csv", new_DSD_fig)
       
  # Convert the modified dictionary back to DataFrame
  updated_data = pd.DataFrame.from_dict(data_dict, orient='index', columns=adjusted_df.columns[1:])
  # Save this updated DataFrame to a new CSV file
  new_csv_path = 'D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/'+FILE_NAME+'_QCtabel.csv'
  updated_data.to_csv(new_csv_path, index_label='TIMESTAMP')
  #new_DSD_fig = new_DSD_fig.tolist
  
  #new_csv_path     

def update_DSD(DSD, arr):
  for i in range (1024):
    y = 32-int(i/32)
    x = int(i%32)+1
    DSD[y][x] = DSD[y][x] + int(arr[3+i])
  return DSD

def write_CSV (output_filename, DSD_array):
  with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(DSD_array)
  


def init_DSD_gif():
  DSD_gif = np.zeros((34,33),int)
  return DSD_gif

def add_tital(DSD):
  H_title = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11.0,13.0,15.0,17.0,19.0,21.5,24.5]
  V_title = [20.8,17.6,15.2,13.6,12.0,10.4,8.8,7.6,6.8,6.0,5.2,4.4,3.8,3.4,3.0,2.6,2.2,1.9,1.7,1.5,1.3,1.1,0.95,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05]
  for i in range (0,32):
    DSD[0][i+1]  = H_title[i]
    DSD[i+1][0]  = V_title[i]
  return DSD





change_data_format()
Quality_ctrl()
