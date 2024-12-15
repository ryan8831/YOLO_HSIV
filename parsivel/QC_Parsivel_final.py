import sys
import getpass
import pandas as pd
import numpy as np
from QC_utils import sum_hr_info, update_SV_o,init_SV_gif,add_tital,write_CSV
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
# load .mat file
from scipy.io import loadmat

# ===========================================
# user conf.
TIME_SEL   = "18:00"
MIN_OUT_EN = 0
CHANGE_EN  = 1
PATH_ROOT = "D:/Raindrop_folder/Rainfall_project_2023/Parsivel_data/"
FILE_NAME = "Parsivel_20240331-18TH"

# ============================================================================================================================================================================
aD=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5 ,7.5 ,8.5 ,9.5 ,11,13,15,17,19,21,24.5])
dD=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,3,3,3,])
aV=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.10,1.30,1.50,1.70,1.90,2.20,2.60,3,3.40,3.80,4.40,5.20,6,6.80,7.60,8.80,10.4,12,13.6,15.2,17.6,20.8,])
dV=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,])
sTIME      = 0
rINTENSITY = 1
nPARTICLES = 2

def change_data_format (enable=1):
  if (enable):
    file1_path = PATH_ROOT+FILE_NAME+'.csv'
    columns_to_keep = [
        'TIMESTAMP', 'sTime', 'rIntensity', 'nParticles'
    ] + [f'drop({i},{j})' for i in range(1, 33) for j in range(1, 33)]
    file1_df_reloaded = pd.read_csv(file1_path, skiprows=[0, 2, 3])
    filtered_columns = [col for col in file1_df_reloaded.columns if col in columns_to_keep]
    file1_adjusted_final = file1_df_reloaded[filtered_columns]
    adjusted_file_path = PATH_ROOT+ FILE_NAME +'_ORI.csv'
    file1_adjusted_final.to_csv(adjusted_file_path, index=False)
    print(f"Adjusted file saved to: {adjusted_file_path}")
#=====================================================================================================================================
def get_QC_Vt_fc(aD,aV,C=0.4):
    CDV = np.zeros(1024)
    QV  = np.zeros(len(aD)); 
    PQV = np.zeros(len(aD)); NQV = np.zeros(len(aD))
    #QV  = 9.65-10.3*np.exp(-0.6*aD)
    QV  = -0.1021+4.932*aD-0.9551*(aD**2)+0.07934*(aD**3)-0.002362*(aD**4)
    PQV = QV*(1+C)
    NQV = QV*(1-C)
    for n in range(1024):
        class_v = int(n/32)
        class_d = int(n%32)
        if (class_d > 25 or class_d < 2):
          continue
        if (PQV[class_d] < 0 or NQV[class_d]<0):
          continue
        max_idx = np.argmax(np.argwhere(aV<PQV[class_d]))
        min_idx = np.argmax(np.argwhere(aV<NQV[class_d])) + 1
        if (class_v >= min_idx and class_v <= max_idx):
          CDV[n] = 1
    return np.asanyarray(CDV,dtype=bool)
############################################
# this subroutine is calculate NDij (n[time],32[aV],32[aD]) from diameter and velocity spectrum (SV)
def SV2NDij_fc(values,aD,dD,aV,mn=60):
    SV = values[3:].copy()
    NDij=np.zeros(1024)
    for n in range(1024):
      i = int(n/32)
      j = int(n%32)                              # j代表粒徑index
      seff=180*(30-aD[j]/2)/1000/1000
      # seff=0.0054
      NDij[n]=(1/(seff*mn*dD[j]))*(SV[n]/aV[i])
    return NDij
#--------------------------
# this subroutine is calculate R with reference velocity from equation
def NDij2CR_fc(values,NDij,aD,dD,aV,mn=60):
    num_data=NDij.shape[0]
    CRij = 0
    R_temp=0
    for n in range(1024):
      i = int(n/32)
      j = int(n%32)                              # j代表粒徑index
      R_temp+=6*np.pi*10**(-4)*((aV[i]*aD[j]**3*NDij[n]*dD[j]))
    CRij = R_temp
    values[rINTENSITY] = CRij
    return values,CRij
########################################################################################################################################################################################################################################################################
#%%
def Quality_ctrl ():
  adjusted_df = pd.read_csv(PATH_ROOT+ FILE_NAME +'_ORI.csv') # 將DataFrame轉換為字典
  # QC SV data
  data_dict = adjusted_df.set_index('TIMESTAMP').T.to_dict('list')
  data_dict_QC = data_dict.copy() 
  # 顯示字典中的前幾個項目來確認結果
  new_SV_fig = init_SV_gif()
  # ------------------------------------------------------------------------------
  # 1. remove the total number of particles < 10
  # 2. remove the Ri < R_threshold
  for key, values in data_dict_QC.items():
      if (values[nPARTICLES]<10):
        values[nPARTICLES] = 0
        for k in range (3,1027,1) :  values[k] = 0
      if(values[rINTENSITY] < 0.1):
        values[nPARTICLES] = 0
        values[rINTENSITY] = 0
        for k in range (3,1027,1) :  values[k] = 0
  # 3. remove the SV from SV mask (0.5 standard deviation of terminal velocity)
  SV_mask=get_QC_Vt_fc(aD,aV,C=0.5)
  # 4. remove diameter < 0.2mm and > 10mm
  for key, values in data_dict_QC.items():
    for i in range (1024):
      if SV_mask[i] :
       continue
      else :
       values[i+3]=0
  # ------------------------------------------------------------------------------
  # convert the SPECTRUM (SV) to DSD number concentration (NDij)
  # calculate the CR from NDij
  # 1. without QC
    pass   # todo
  # 2. with QC
  for key, values in data_dict_QC.items():
    NDij_QC=SV2NDij_fc(values,aD,dD,aV)
    values, CRij=NDij2CR_fc(values,NDij_QC,aD,dD,aV)
  # ------------------------------------------------------------------------------
  # SV表格輸出
  for key, values in data_dict.items():
    new_SV_fig = update_SV_o(new_SV_fig, values)
    new_SV_fig = np.asanyarray(new_SV_fig,float)
    new_SV_fig = add_tital(new_SV_fig)

  plot_all_DSD(new_SV_fig)

  # ------------------------------------------------------------------------------
  sum_data = np.zeros([32,4])
  for key, values in data_dict.items():
    sum_data = sum_hr_info(values, sum_data)
  # ------------------------------------------------------------------------------
  write_CSV (PATH_ROOT+FILE_NAME+"_SV_QC.csv", new_SV_fig)
  write_CSV (PATH_ROOT+FILE_NAME+"_SUM_QC.csv",sum_data )

  # Convert the modified dictionary back to DataFrame
  data_dict_QC_copy = data_dict_QC.copy()
  updated_data = pd.DataFrame.from_dict(data_dict_QC_copy, orient='index', columns=adjusted_df.columns[1:])
  # Save this updated DataFrame to a new CSV file
  new_csv_path = PATH_ROOT+FILE_NAME+'_TABLE_QC.csv'
  updated_data.to_csv(new_csv_path, index_label='TIMESTAMP')

  return data_dict_QC
############################################
def min_SV(dict_, time_sel, enable):
  if (enable) :
    MIN_SV_fig = init_SV_gif()
    for key, values in dict_.items():
      time = values[0][:5]
      if (time != time_sel):
        continue
      else :
        min_values = values
        break

    MIN_SV_fig = update_SV_o(MIN_SV_fig, min_values)
    MIN_SV_fig = np.asanyarray(MIN_SV_fig,float)
    MIN_SV_fig = add_tital(MIN_SV_fig)
    write_CSV (PATH_ROOT+FILE_NAME+"_SV_"+time_sel[:2]+time_sel[3:]+"_QC.csv", MIN_SV_fig)

    plot_min_DSD(MIN_SV_fig, time_sel)
############################################
def plot_min_DSD(min_SV_table,time_sel):
  class_sum = np.zeros(32)  
  class_sum_log = np.zeros(32)
  for i in range (1,33,1):
    for j in range (1,33,1):
      class_sum[i-1] += min_SV_table[j][i] 
  for i in range (32):
    if(class_sum[i]==0): continue
    class_sum_log[i] = math.log10(class_sum[i])

  print(class_sum)
  x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
  #x=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,3,3,3,]
  plt.title("MIN:"+time_sel[:2]+time_sel[3:]+" DSD")
  plt.xlabel("Diameter class")
  plt.ylabel('Log10  Number of drops')
  plt.bar(x, height=class_sum_log, width=0.6)
  plt.savefig(PATH_ROOT+FILE_NAME+"_SV_"+time_sel[:2]+time_sel[3:]+"_MIN_DSD_QC.png", dpi=300)
  plt.show()

def plot_all_DSD(all_SV_table):
  class_sum = np.zeros(32)
  class_sum_log = np.zeros(32)
  for i in range (1,33,1):
    for j in range (1,33,1):
      class_sum[i-1] += all_SV_table[j][i]
  for i in range (32):
    if(class_sum[i]==0): continue
    class_sum_log[i] = math.log10(class_sum[i])

  print(class_sum)
  x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
  plt.title(FILE_NAME+" DSD")
  plt.xlabel("Diameter class")
  plt.ylabel('Log10  Number of drops')
  plt.bar(x, height=class_sum_log, width=0.6)
  plt.savefig(PATH_ROOT+FILE_NAME+"_SV_DSD_QC.png", dpi=300)
  plt.show()

# ============================================================================================================================================================================================ #
#                                                                                                                                                                                              #
#   MMMMMMMM               MMMMMMMM                    AAA                    IIIIIIIIII      NNNNNNNN        NNNNNNNN                                                                         #
#   M:::::::M             M:::::::M                   A:::A                   I::::::::I      N:::::::N       N::::::N                                                                         #
#   M::::::::M           M::::::::M                  A:::::A                  I::::::::I      N::::::::N      N::::::N                                                                         #
#   M:::::::::M         M:::::::::M                 A:::::::A                 II::::::II      N:::::::::N     N::::::N                                                                         #
#   M::::::::::M       M::::::::::M                A:::::::::A                  I::::I        N::::::::::N    N::::::N                                                                         #
#   M:::::::::::M     M:::::::::::M               A:::::A:::::A                 I::::I        N:::::::::::N   N::::::N                                                                         #
#   M:::::::M::::M   M::::M:::::::M              A:::::A A:::::A                I::::I        N:::::::N::::N  N::::::N                                                                         #
#   M::::::M M::::M M::::M M::::::M             A:::::A   A:::::A               I::::I        N::::::N N::::N N::::::N                                                                         #
#   M::::::M  M::::M::::M  M::::::M            A:::::A     A:::::A              I::::I        N::::::N  N::::N:::::::N                                                                         #
#   M::::::M   M:::::::M   M::::::M           A:::::AAAAAAAAA:::::A             I::::I        N::::::N   N:::::::::::N                                                                         #
#   M::::::M    M:::::M    M::::::M          A:::::::::::::::::::::A            I::::I        N::::::N    N::::::::::N                                                                         #
#   M::::::M     MMMMM     M::::::M         A:::::AAAAAAAAAAAAA:::::A           I::::I        N::::::N     N:::::::::N                                                                         #
#   M::::::M               M::::::M        A:::::A             A:::::A        II::::::II      N::::::N      N::::::::N                                                                         #
#   M::::::M               M::::::M       A:::::A               A:::::A       I::::::::I      N::::::N       N:::::::N                                                                         #
#   M::::::M               M::::::M      A:::::A                 A:::::A      I::::::::I      N::::::N        N::::::N                                                                         #
#   MMMMMMMM               MMMMMMMM     AAAAAAA                   AAAAAAA     IIIIIIIIII      NNNNNNNN         NNNNNNN                                                                         #
#                                                                                                                                                                                              #
# ============================================================================================================================================================================================ #

change_data_format(CHANGE_EN)
data_dict_QC =  Quality_ctrl()
min_SV(data_dict_QC, TIME_SEL, MIN_OUT_EN)