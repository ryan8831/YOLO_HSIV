import matplotlib.pyplot as plt 
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import math as m
from utils import csv_to_xlsx_pd, fnc_show_img
from path_def import draw_path_def


### ========
# PATH DEFINE
ID, WRITE_PATH, WRITE_PATH_ANALYZE, WRITE_PATH_ARRIMG = draw_path_def()


### ========
excel1='雨滴每秒結果' + ID
excel2='雨滴小時結果' + ID
excel3='雨滴分鐘結果' + ID


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
Y=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23 ,26]   #速度 Y
#X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 ,3.2 ,3.6 ,4 ,4.8 ,5.6 ,6.4 ,7.8 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 ,3 ,3.1 , 3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4 ,4.8,5.6 ,6.4 ,7.8] #直徑 X
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
# print(9.65-10.3*np.exp(-0.6*X.astype(float)))
for i in range(0,len(tmpdatahsum2)):
    #f, ax = plt.subplots(figsize=(10, 6))
    f, ax = plt.subplots(figsize=(20, 6))  #讓輸出圖片變長
    plt.tick_params(axis='x', labelsize=8) #讓標籤數字變小
    # sns.heatmap(df, annot=False,cmap='YlOrRd',ax=ax)
    # ax.grid()

    c = ax.pcolormesh(X, Y, np.array(tmpdatahsum2[i]), cmap='YlOrRd')
    f.colorbar(c)
    plt.plot(X,(9.65-10.3*np.exp(-0.6*X.astype(float))),label='F(x)=9.65-10.3*e(-0.6*x)')
    # x1=np.arange(0,10,0.5)
    # y1=np.arange(0,12,0.5)
    

    plt.title('number & velocity & diameter ({}H).png'.format(i+strat_index))
    plt.xlabel( 'diameter' )
    plt.ylabel('velocity')
    plt.legend()
    #plt.xticks([0.1,0.4,0.7,1.0,1.6,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8])#室內
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.4,3.6]) #室外
    plt.yticks([0.125,0.5,0.875,1.25,1.5,1.75,2,2.25,3,3.5,4,4.5,5,5.5,6,6.5,7,10,10.5,11,11.5,12,12.5,13])
    plt.xlim(0,6)
    plt.ylim(0,13)
    plt.grid()
    # plt.show()
    plt.savefig(WRITE_PATH_ANALYZE + ID + '_雨量&速度&直徑(第{}小時).png'.format(i+strat_index))
# ------------------------------------------heatmap