import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import copy
import math
import csv
import numpy as np
import pandas as pd
import math as m
from km_utils import csv_to_xlsx_pd, fnc_show_img
from path_def import draw_path_def


### ========
# PATH DEFINE
ID, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG, MASK_EN = draw_path_def()


# ================================================
aD=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5 ,7.5 ,8.5 ,9.5 ,11,13,15,17,19,21,24.5])
dD=np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,2,2,2,2,3,3,3,])
aV=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.10,1.30,1.50,1.70,1.90,2.20,2.60,3,3.40,3.80,4.40,5.20,6,6.80,7.60,8.80,10.4,12,13.6,15.2,17.6,20.8,])
dV=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,1.6,1.6,1.6,1.6,3.2,3.2,])
SV_mask_=np.zeros((32,32))
def get_QC_Vt_fc(aD,aV,C=0.4):
    CDV=np.zeros([len(aD),len(aV)])
    #QV=-0.1021+4.932*aD-0.9551*(aD**2)+0.07934*(aD**3)-0.002362*(aD**4)
    QV=9.65-10.3*np.exp(-0.6*aD.astype(float))
    PQV=QV*(1+C)
    NQV=QV*(1-C)
    for n in range(len(aD)):
        ijk=np.where((aV>NQV[n]) & (aV<PQV[n]))
        CDV[ijk,n]=1
    return CDV,QV,PQV,NQV

C=0.5 #[follow the paper from CWA] 
SV_mask=get_QC_Vt_fc(aD,aV,C=C)[0]
SV_mask_=get_QC_Vt_fc(aD,aV,C=C)[0]
# 4. remove diameter < 0.2mm and > 10mm
#SV_mask[:,0:np.where(aD>0.2)[0][0]]=0
#SV_mask[:,np.where(aD>10)[0][0]:]=0
for i in range (32):
    SV_mask_[i] = SV_mask[31-i]

### ========
excel1='雨滴每秒結果' + ID
excel2='雨滴小時結果' + ID
excel3='雨滴分鐘結果' + ID
excel4='雨滴所有顆粒' + ID
exe4 = False

with open( WRITE_PATH_ANALYZE + '雨滴分鐘結果' + ID + '.csv', newline='',encoding="utf-8") as csv1:
    rain_minute = list(csv.reader(csv1))
    rain_minute=np.array(rain_minute)
    rain_minute=rain_minute[1:]

with open( WRITE_PATH_ANALYZE + '雨滴小時結果' + ID + '.csv', newline='',encoding="utf-8") as csv2:
    rain_hours = list(csv.reader(csv2))
    rain_hours=np.array(rain_hours)
    rain_hours=rain_hours[1:]

with open( WRITE_PATH_ANALYZE + '雨滴每秒結果' + ID + '.csv', newline='',encoding="utf-8") as csv3:
    rain_sec = list(csv.reader(csv3))
    rain_sec=np.array(rain_sec)
    rain_sec=rain_sec[1:]

if exe4:
  with open( WRITE_PATH_ANALYZE+ '雨滴所有顆粒' + ID + '.csv', newline='',encoding="utf-8") as csv4:
      rain_sec_all = list(csv.reader(csv4))
      rain_sec_all =np.array(rain_sec_all)
      rain_sec_all =rain_sec_all[1:]

print(rain_sec[0][0][11:13])  #小時
print(rain_sec[0][0][13:15])  #分鐘
print(rain_sec[0][0][15:17])  #秒
strat_index=int(rain_sec[0][0][11:13])
end_index=int(float(rain_sec[-1][0][11:13]))
tmp_rainfal_min=[]
tmp_rainfal_minh=[]
for i in range(0,len(rain_minute)-1):
    if rain_minute[i][0][2:]!=rain_minute[i+1][0][2:]: #儲存每分鐘雨量&數量
        tmp_rainfal_min.append(rain_minute[i][0:3])
        if rain_minute[i][0][:2]!=rain_minute[i+1][0][:2]:  #儲存每小時雨量&數量
            tmp_rainfal_minh.append(np.array(tmp_rainfal_min))
            #print(np.array(tmp_rainfal_min))
            tmp_rainfal_min=[]
tmp_rainfal_minh=np.array(tmp_rainfal_minh)
print("shape =" , tmp_rainfal_minh.shape)
# print(tmp_rainfal_minh[0])

# -----------------------------------------------------圖片儲存
for i in range(0,len(tmp_rainfal_minh)): #tmp_rainfal_minh: 時間 雨量 數量
    X=tmp_rainfal_minh[i][:,0]       #取出幾點幾分 EX:'0901'
    Y=list(map(int,tmp_rainfal_minh[i][:,2])) #取出每分鐘累積數量
    # print(tmp_rainfal_min[:,0])
    plt.figure().set_size_inches(20,8)
    plt.bar(X,Y)
    plt.xticks( X , rotation=45)
    # plt.yticks( [0,300] , rotation=45)
    plt.title("min vs number ({} hour)".format(i+strat_index))
    plt.savefig(WRITE_PATH_ANALYZE + ID +'_數量(第{}小時).png'.format(i+strat_index))

    Y=list(map(float,tmp_rainfal_minh[i][:,1]))
    plt.figure().set_size_inches(20,8)
    plt.bar(X,Y)
    plt.xticks( X , rotation=45)
    # plt.yticks( [0,300] , rotation=45)
    plt.title("min vs mm ({} hour)".format(i+strat_index))
    plt.savefig(WRITE_PATH_ANALYZE + ID + '_雨量(第{}小時).png'.format(i+strat_index))
# -----------------------------------------------------

print(rain_sec[0][8]) #速度 Y
print(rain_sec[0][9]) #直徑 X
print(int(rain_sec[0][0][11:13]))  #小時
#X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 ,3.2 ,3.6 ,4 ,4.8 ,5.6 ,6.4 ,7.8 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
#X=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23]   #直徑 X
#Y=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 7.8, 8, 9.6, 11.2, 12.8, 14.4, 16] #速度 Y
X=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23 ,26]   
Y=[0 ,0.1 ,0.2 ,0.3	,0.4 ,0.5 ,0.6 ,0.7	,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6	,1.8 ,2	,2.4 ,2.8 ,3.2 ,3.6	,4 ,4.8	,5.6 ,6.4 ,7.2 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] 

print(len(X))
print(len(Y))
tmpdatah=np.zeros((end_index-strat_index+1,32,32))    #[小時][Y][X]
tmpdatah2=np.zeros((end_index-strat_index+1,32,32))    #[小時][Y][X]
# print(tmpdatah)
for i in range(0,len(Y)-1):                   #Y 正向
    for j in range(0,len(rain_sec)-1):        #sec數量     
        if Y[i]<float(rain_sec[j][8])<=Y[i+1]:
            for l in range(0,len(X)-1):   #X
                if X[l]<float(rain_sec[j][9])<=X[l+1]:
                    tmpdatah[int(rain_sec[j][0][11:13])-strat_index][31-i][l]=tmpdatah[int(rain_sec[j][0][11:13])-strat_index][31-i][l]+1
                    tmpdatah2[int(rain_sec[j][0][11:13])-strat_index][i][l]=tmpdatah2[int(rain_sec[j][0][11:13])-strat_index][i][l]+1
print(tmpdatah[0])

tmpdatahsum=np.zeros((end_index-strat_index+1,32,32))
tmpdatahsum2=np.zeros((end_index-strat_index+1,32,32))
for i in range(1,len(tmpdatah)+1):
    for j in range(0,i):
        tmpdatahsum[i-1]=tmpdatahsum[i-1]+tmpdatah[j]
        tmpdatahsum2[i-1]=tmpdatahsum2[i-1]+tmpdatah2[j]
tmpdatahsum=tmpdatahsum*30
tmpdatahsum2=tmpdatahsum2*30

Y1=Y[::-1]
# ------------------------------------------csv
for i in range(0,len(tmpdatah)):
    #with open('速度對直徑0616/速度對直徑H{}.csv'.format(i+strat_index),'w',newline='',encoding="utf-8") as f1:
    with open(WRITE_PATH_ANALYZE+ ID + '_速度對直徑H{}.csv'.format(i+strat_index),'w',newline='',encoding="utf-8") as f1:
        writer=csv.writer(f1)
        writer.writerow(X)
        for j in range(0,len(tmpdatahsum[i])):
            writer.writerow(np.insert(tmpdatahsum[i][j],0,Y1[j]))
        writer.writerow(X)
    #csv_to_xlsx_pd(WRITE_PATH_ANALYZE+ ID +'_速度對直徑H{}'.format(i+strat_index))
    csv_to_xlsx_pd('_速度對直徑H{}'.format(i+strat_index), WRITE_PATH_ANALYZE+ ID)
# ------------------------------------------csv
# Y1=Y[::-1] row
# X           colum

# ------------------------------------------heatmap
X=np.array(X)
Y1=np.array(Y1)
Y=np.array(Y)
x_line = np.linspace(X.min(), X.max(), 1000)  # A smooth line
QV=-0.1021+4.932*x_line-0.9551*(x_line**2)+0.07934*(x_line**3)-0.002362*(x_line**4)
# print(9.65-10.3*np.exp(-0.6*X.astype(float)))
for i in range(0,len(tmpdatahsum2)):
    #f, ax = plt.subplots(figsize=(10, 6))
    f, ax = plt.subplots(figsize=(18, 12))  #讓輸出圖片變長
    plt.tick_params(axis='x', labelsize=20, rotation = 45) #讓標籤數字變小
    plt.tick_params(axis='y', labelsize=20 ) #讓標籤數字變小
    # sns.heatmap(df, annot=False,cmap='YlOrRd',ax=ax)
    # ax.grid()
    if MASK_EN : tmpdatahsum2[i][SV_mask== 0] = 0
    tmpdatahsum2[i][tmpdatahsum2[i]==0] = np.nan
    #c = ax.pcolormesh(X, Y, np.array(tmpdatahsum2[i]), cmap='YlOrRd')
    c = ax.pcolormesh(X, Y, np.array(tmpdatahsum2[i]), cmap='viridis')
    f.colorbar(c)
    plt.plot(X,(9.65-10.3*np.exp(-0.6*X.astype(float))),label='F(x)=9.65-10.3*e(-0.6*x)')
    
    #ax.plot(x_line,QV,'k-',color='blue',zorder=3,label='ref. V$_t$')
    plt.plot(x_line,QV,'k-',color='red',zorder=3,label='-0.1021+4.932*x-0.9551*(x**2)+0.07934*(x**3)-0.002362*(x**4)')
    # x1=np.arange(0,10,0.5)
    # y1=np.arange(0,12,0.5)
    

    plt.title(ID+'\n number & velocity & diameter '.format(i+strat_index), size=20)
    plt.xlabel( 'diameter' )
    plt.ylabel('velocity')
    plt.legend()
    #plt.xticks([0.1,0.4,0.7,1.0,1.6,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8])#室內
    #plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.4,3.6]) #室外
    #plt.yticks([0.125,0.5,0.875,1.25,1.5,1.75,2,2.25,3,3.5,4,4.5,5,5.5,6,6.5,7,10,10.5,11,11.5,12,12.5,13])
    #plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2,14.4])
    #plt.xticks([0.125,0.25,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10])
    plt.yticks([0.1,  0.5, 1, 1.5 , 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2,14.4])
    plt.xticks([0,0.125,0.25,0.5 ,0.75  ,1,1.25  ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10])
    plt.xlim(0,5)
    plt.ylim(0,12)
    plt.grid()
    #plt.show()
    plt.savefig(WRITE_PATH_ANALYZE + ID + '_雨量&速度&直徑(第{}小時).png'.format(i+strat_index))


if exe4:
  # add
  strat_index_all=int(rain_sec_all[0][0][11:13])
  end_index_all=int(float(rain_sec_all[-1][0][11:13]))
  #X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 ,3.2 ,3.6 ,4 ,4.8 ,5.6 ,6.4 ,7.8 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
  #Y=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23]   #直徑 X
  #X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 7.8, 8, 9.6, 11.2, 12.8, 14.4, 16] #速度 Y
  X=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23 ,26]   #速度 Y
  Y=[0 ,0.1 ,0.2 ,0.3	,0.4 ,0.5 ,0.6 ,0.7	,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6	,1.8 ,2	,2.4 ,2.8 ,3.2 ,3.6	,4 ,4.8	,5.6 ,6.4 ,7.2 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
  
  print(len(X))
  print(len(Y))
  tmpdatah=np.zeros((end_index_all-strat_index_all+1,32,32))    #[小時][Y][X]
  tmpdatah2=np.zeros((end_index_all-strat_index_all+1,32,32))    #[小時][Y][X]
  # print(tmpdatah)
  for i in range(0,len(Y)-1):                   #Y 正向
      for j in range(0,len(rain_sec_all)-1):        #sec數量
          if Y[i]<float(rain_sec_all[j][8])<=Y[i+1]:
              for l in range(0,len(X)-1):   #X
                  if X[l]<float(rain_sec_all[j][9])<=X[l+1]:
                      tmpdatah[int(rain_sec_all[j][0][11:13])-strat_index_all][31-i][l]=tmpdatah[int(rain_sec_all[j][0][11:13])-strat_index_all][31-i][l]+1
                      tmpdatah2[int(rain_sec_all[j][0][11:13])-strat_index_all][i][l]=tmpdatah2[int(rain_sec_all[j][0][11:13])-strat_index_all][i][l]+1
  print(tmpdatah[0])
  
  tmpdatahsum=np.zeros((end_index_all-strat_index_all+1,32,32))
  tmpdatahsum2=np.zeros((end_index_all-strat_index_all+1,32,32))
  for i in range(1,len(tmpdatah)+1):
      for j in range(0,i):
          tmpdatahsum[i-1]=tmpdatahsum[i-1]+tmpdatah[j]
          tmpdatahsum2[i-1]=tmpdatahsum2[i-1]+tmpdatah2[j]
  tmpdatahsum=tmpdatahsum*30
  tmpdatahsum2=tmpdatahsum2*30
  
  Y1=Y[::-1]
  # ------------------------------------------csv
  for i in range(0,len(tmpdatah)):
      #with open('速度對直徑0616/速度對直徑H{}.csv'.format(i+strat_index_all),'w',newline='',encoding="utf-8") as f1:
      with open(WRITE_PATH_ANALYZE+ ID + '_速度對直徑H{}_all.csv'.format(i+strat_index_all),'w',newline='',encoding="utf-8") as f1:
          writer=csv.writer(f1)
          writer.writerow(X)
          for j in range(0,len(tmpdatahsum[i])):
              writer.writerow(np.insert(tmpdatahsum[i][j],0,Y1[j]))
          writer.writerow(X)
      #csv_to_xlsx_pd(WRITE_PATH_MODIFY+ ID +'_速度對直徑H{}'.format(i+strat_index_all))
      csv_to_xlsx_pd('_速度對直徑H{}_all'.format(i+strat_index_all), WRITE_PATH_ANALYZE+ ID)
  # ------------------------------------------csv
  # Y1=Y[::-1] row
  # X           colum
  
  # ------------------------------------------heatmap
  X=np.array(X)
  Y1=np.array(Y1)
  Y=np.array(Y)
  # print(9.65-10.3*np.exp(-0.6*X.astype(float)))
  for i in range(0,len(tmpdatahsum2)):
      #f, ax = plt.subplots(figsize=(10, 6))
      f, ax = plt.subplots(figsize=(18, 12))  #讓輸出圖片變長
      plt.tick_params(axis='x', labelsize=20) #讓標籤數字變小
      plt.tick_params(axis='y', labelsize=20) #讓標籤數字變小
      # sns.heatmap(df, annot=False,cmap='YlOrRd',ax=ax)
      # ax.grid()
  
      c = ax.pcolormesh(X, Y, np.array(tmpdatahsum2[i]), cmap='viridis')
      f.colorbar(c)
      plt.plot(X,(9.65-10.3*np.exp(-0.6*X.astype(float))),label='F(x)=9.65-10.3*e(-0.6*x)')
      # x1=np.arange(0,10,0.5)
      # y1=np.arange(0,12,0.5)
  
  
      plt.title(ID+'\n number & velocity & diameter ({}H)_all.png'.format(i+strat_index_all), size=20)
      plt.xlabel( 'diameter' )
      plt.ylabel('velocity')
      plt.legend()
      #plt.xticks([0.1,0.4,0.7,1.0,1.6,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8])#室內
      #plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.4,3.6]) #室外
      #plt.yticks([0.125,0.5,0.875,1.25,1.5,1.75,2,2.25,3,3.5,4,4.5,5,5.5,6,6.5,7,10,10.5,11,11.5,12,12.5,13])
      #plt.xticks([0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23])
      #plt.yticks([0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 6.8, 7.6, 8.4, 8.8, 9.6, 11.2, 12.8, 14.4, 16, 19.2, 22.4 ]) #室外
      plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2])
      plt.xticks([0.125,0.25,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10])
  
      plt.xlim(0,6)
      plt.ylim(0,13)
      plt.grid()
      # plt.show()
      plt.savefig(WRITE_PATH_ANALYZE + ID + '_雨量&速度&直徑(第{}小時)_all.png'.format(i+strat_index_all))
  # ------------------------------------------heatmap














# ------------------------------------------heatmap
#DSD pic
df = pd.read_csv(WRITE_PATH_ANALYZE + '雨滴每秒結果' + ID + '.csv')

# Define the classification conditions based on the '直徑' values
bins = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, float('inf')]
labels = range(1, 34)

#計算每個雨滴的雨量
df['雨量'] = (df['長軸'] ** 2 * 8.13e-2 * 8.13e-2 * df['短軸'] * 8.13e-2 * np.pi / 6 * 30 / 75.6 / 52.5)
df['累積雨量'] = df['雨量'].cumsum()

# 將'直徑'資料轉成class
df['Dclass'] = pd.cut(df['直徑'], bins=bins, labels=labels, right=False)

# Count the number of occurrences in each class
class_counts = df['Dclass'].value_counts().sort_index()*30

# 每個class的雨量
rainfall_sum = df.groupby('Dclass', observed=False)['雨量'].sum()

#每個class的雨量占總雨量百分比
total_rainfall = df['雨量'].sum()
rainfall_to_percentage = (rainfall_sum / total_rainfall) * 100


results_df = pd.DataFrame({
    'Dclass': class_counts.index,
    'count': class_counts.values,
    'rainfall_sum': rainfall_sum,
    'rainfall_to_percentage': rainfall_to_percentage
})

results_df = results_df[:-1]

results_df=results_df.transpose()

dclass=results_df.iloc[0, :]
count = results_df.iloc[1,:]
rainfall = results_df.iloc[2,:]
count_np = count.copy()
count_np = np.asanyarray(count_np, dtype=float)
for i in range (32):
    count_np[i] = math.log10(count_np[i]) if (count_np[i]!=0) else 0

x_label = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
plt.figure(figsize=(18, 12))  # Adjust the figure size

bars = plt.bar(dclass, count_np, color='blue')

#讓長條圖上有y數值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '%.2f' % float(height), ha='center', va='bottom')


plt.xlabel('DIAMETER class (mm) ')
plt.ylabel('count (log10)')
plt.xticks(x_label)
plt.title(ID + 'DSD')

#save file
#results_df.to_excel(WRITE_PATH_ANALYZE + ID + "_log10_DSD.xlsx", header=False)
plt.savefig(WRITE_PATH_ANALYZE + ID  + "_log10_DSD.png")
#plt.show()
plt.close()

plt.figure(figsize=(18, 12))  # Adjust the figure size
bars = plt.bar(dclass, count, color='blue')
#讓長條圖上有y數值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '%d' % int(height), ha='center', va='bottom')

plt.xlabel('DIAMETER class (mm)')
plt.ylabel('count')
plt.xticks(x_label)
plt.title( ID + ' DSD')

#save file
results_df.to_excel(WRITE_PATH_ANALYZE + ID + "_DSD.xlsx", header=False)
plt.savefig(WRITE_PATH_ANALYZE + ID + "_DSD.png")
plt.close()

#--------雨量作圖--------------------

plt.figure(figsize=(18, 12))  # Adjust the figure size
bars = plt.bar(dclass, rainfall, color='blue')
#讓長條圖上有y數值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '%.2f' % float(height), ha='center', va='bottom')


plt.xlabel('DIAMETER class (mm)')
plt.ylabel('rainfall(mm)')
plt.xticks(x_label)
plt.title( ID + ' DSD_rainfall')

#save file
df.to_excel(WRITE_PATH_ANALYZE + ID + "_update_data.xlsx", index=False)
plt.savefig(WRITE_PATH_ANALYZE + ID + "_DSD_rainfall.png")
plt.close()