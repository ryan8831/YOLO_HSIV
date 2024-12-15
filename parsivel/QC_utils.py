import csv
import matplotlib
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt 
from datetime            import datetime, timedelta
from scipy.interpolate   import griddata

def init_SV_gif():
  SV_gif = np.zeros((34,33),int)
  return SV_gif

def update_SV_o(SV, arr):
  for i in range (1024):
    y = 32-int(i/32)
    x = int(i%32)+1
    SV[y][x] = SV[y][x] + int(arr[2+i])
  return SV

def add_tital(SV):
  H_title = [0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11.0,13.0,15.0,17.0,19.0,21.5,24.5]
  V_title = [20.8,17.6,15.2,13.6,12.0,10.4,8.8,7.6,6.8,6.0,5.2,4.4,3.8,3.4,3.0,2.6,2.2,1.9,1.7,1.5,1.3,1.1,0.95,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05]
  for i in range (0,32):
    SV[0][i+1]  = H_title[i]
    SV[i+1][0]  = V_title[i]
  return SV

def write_CSV (output_filename, DSD_array):
  with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(DSD_array)

def sum_hr_info(key,values, sum_data):
  date = datetime.strptime(str(key), '%Y/%m/%d %H:%M:%S')
  formatted_date = date.strftime('%Y/%m/%d %H:%M:%S')
  hr = int(formatted_date[11:13])
  if (hr == 0):
      sum_data[0][0] = 0
      sum_data[0][1] = sum_data[0][1] + values[1]
      sum_data[0][2] = sum_data[0][2] + values[0]
      sum_data[0][3] = sum_data[0][1] / 60
  elif (hr == 1):
      sum_data[1][0] = 1
      sum_data[1][1] = sum_data[1][1] + values[1]
      sum_data[1][2] = sum_data[1][2] + values[0]
      sum_data[1][3] = sum_data[1][1] / 60
  elif (hr == 2):
      sum_data[2][0] = 2
      sum_data[2][1] = sum_data[2][1] + values[1]
      sum_data[2][2] = sum_data[2][2] + values[0]
      sum_data[2][3] = sum_data[2][1] / 60
  elif (hr == 3):
      sum_data[3][0] = 3
      sum_data[3][1] = sum_data[3][1] + values[1]
      sum_data[3][2] = sum_data[3][2] + values[0]
      sum_data[3][3] = sum_data[3][1] / 60
  elif (hr == 4):
      sum_data[4][0] = 4
      sum_data[4][1] = sum_data[4][1] + values[1]
      sum_data[4][2] = sum_data[4][2] + values[0]
      sum_data[4][3] = sum_data[4][1] / 60
  elif (hr == 5):
      sum_data[5][0] = 5
      sum_data[5][1] = sum_data[5][1] + values[1]
      sum_data[5][2] = sum_data[5][2] + values[0]
      sum_data[5][3] = sum_data[5][1] / 60
  elif (hr == 6):
      sum_data[6][0] = 6
      sum_data[6][1] = sum_data[6][1] + values[1]
      sum_data[6][2] = sum_data[6][2] + values[0]
      sum_data[6][3] = sum_data[6][1] / 60
  elif (hr == 7):
      sum_data[7][0] = 7
      sum_data[7][1] = sum_data[7][1] + values[1]
      sum_data[7][2] = sum_data[7][2] + values[0]
      sum_data[7][3] = sum_data[7][1] / 60
  elif (hr == 8):
      sum_data[8][0] = 8
      sum_data[8][1] = sum_data[8][1] + values[1]
      sum_data[8][2] = sum_data[8][2] + values[0]
      sum_data[8][3] = sum_data[8][1] / 60
  elif (hr == 9):
      sum_data[9][0] = 9
      sum_data[9][1] = sum_data[9][1] + values[1]
      sum_data[9][2] = sum_data[9][2] + values[0]
      sum_data[9][3] = sum_data[9][1] / 60
  elif (hr == 10):
      sum_data[10][0] = 10
      sum_data[10][1] = sum_data[10][1] + values[1]
      sum_data[10][2] = sum_data[10][2] + values[0]
      sum_data[10][3] = sum_data[10][1] / 60
  elif (hr == 11):
      sum_data[11][0] = 11
      sum_data[11][1] = sum_data[11][1] + values[1]
      sum_data[11][2] = sum_data[11][2] + values[0]
      sum_data[11][3] = sum_data[11][1] / 60
  elif (hr == 12):
      sum_data[12][0] = 12
      sum_data[12][1] = sum_data[12][1] + values[1]
      sum_data[12][2] = sum_data[12][2] + values[0]
      sum_data[12][3] = sum_data[12][1] / 60
  elif (hr == 13):
      sum_data[13][0] = 13
      sum_data[13][1] = sum_data[13][1] + values[1]
      sum_data[13][2] = sum_data[13][2] + values[0]
      sum_data[13][3] = sum_data[13][1] / 60
  elif (hr == 14):
      sum_data[14][0] = 14
      sum_data[14][1] = sum_data[14][1] + values[1]
      sum_data[14][2] = sum_data[14][2] + values[0]
      sum_data[14][3] = sum_data[14][1] / 60
  elif (hr == 15):
      sum_data[15][0] = 15
      sum_data[15][1] = sum_data[15][1] + values[1]
      sum_data[15][2] = sum_data[15][2] + values[0]
      sum_data[15][3] = sum_data[15][1] / 60
  elif (hr == 16):
      sum_data[16][0] = 16
      sum_data[16][1] = sum_data[16][1] + values[1]
      sum_data[16][2] = sum_data[16][2] + values[0]
      sum_data[16][3] = sum_data[16][1] / 60
  elif (hr == 17):
      sum_data[17][0] = 17
      sum_data[17][1] = sum_data[17][1] + values[1]
      sum_data[17][2] = sum_data[17][2] + values[0]
      sum_data[17][3] = sum_data[17][1] / 60
  elif (hr == 18):
      sum_data[18][0] = 18
      sum_data[18][1] = sum_data[18][1] + values[1]
      sum_data[18][2] = sum_data[18][2] + values[0]
      sum_data[18][3] = sum_data[18][1] / 60
  elif (hr == 19):
      sum_data[19][0] = 19
      sum_data[19][1] = sum_data[19][1] + values[1]
      sum_data[19][2] = sum_data[19][2] + values[0]
      sum_data[19][3] = sum_data[19][1] / 60
  elif (hr == 20):
      sum_data[20][0] = 20
      sum_data[20][1] = sum_data[20][1] + values[1]
      sum_data[20][2] = sum_data[20][2] + values[0]
      sum_data[20][3] = sum_data[20][1] / 60
  elif (hr == 21):
      sum_data[21][0] = 21
      sum_data[21][1] = sum_data[21][1] + values[1]
      sum_data[21][2] = sum_data[21][2] + values[0]
      sum_data[21][3] = sum_data[21][1] / 60
  elif (hr == 22):
      sum_data[22][0] = 22
      sum_data[22][1] = sum_data[22][1] + values[1]
      sum_data[22][2] = sum_data[22][2] + values[0]
      sum_data[22][3] = sum_data[22][1] / 60
  elif (hr == 23):
      sum_data[23][0] = 23
      sum_data[23][1] = sum_data[23][1] + values[1]
      sum_data[23][2] = sum_data[23][2] + values[0]
      sum_data[23][3] = sum_data[23][1] / 60
  return sum_data

def plot_SV_joint_distribution(aD,aV,SV,figurename='SV_joint_distribution.png',title='Joint Distribution'):
    X=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23 ,26]   #速度 Y
    Y=[0 ,0.1 ,0.2 ,0.3	,0.4 ,0.5 ,0.6 ,0.7	,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6	,1.8 ,2	,2.4 ,2.8 ,3.2 ,3.6	,4 ,4.8	,5.6 ,6.4 ,7.2 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
    #-----------------------------------------
    cmap=plt.get_cmap('YlOrRd')
    cmap_colors=cmap(np.linspace(0,1,5000))
    # cmap_colors[0,:]=[1,1,1,0]
    cmap=matplotlib.colors.ListedColormap(cmap_colors)
    #-----------------------------------------
    fig, ax = plt.subplots(figsize=(12,6))
    #im=ax.pcolormesh(aD,aV,SV,cmap=cmap,zorder=2,vmin=0,vmax=0.05)
    im=ax.pcolormesh(X,Y,SV,cmap=cmap,zorder=2)
    # add colormap out of figure ,adjust the figure let size still the same
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.13, 0.05, 0.75])
    cbar=fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Frequency')
    # let cbar scale is percentage
    #cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    # get meanV for reference
    C=0.5
    QV=-0.1021+4.932*aD-0.9551*(aD**2)+0.07934*(aD**3)-0.002362*(aD**4)
    PQV=QV*(1+C)
    NQV=QV*(1-C)
    ax.plot(aD,QV,'k-',zorder=3)
    ax.plot(aD,PQV,'k--',zorder=3)
    ax.plot(aD,NQV,'k--',zorder=3)
    ax.set_xlabel('Diameter [mm]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlim([0,8])
    ax.set_ylim([0,16])
    ax.set_title(title,fontweight='bold')
    #ax.set_xticks([0.125,0.25,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10])
    #ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2])
    ax.legend()
    ax.grid(':',alpha=0.5)
    #fig.patch.set_alpha(0)
    # save the figure
    plt.savefig(figurename,dpi=300)
    #plt.show()
    plt.close()


def gen_Parsivel_SV (rd_path, file_name,hr_file_name,hr_mode):
  # Reload the data with proper headers
  rd_path = rd_path + file_name + "/"
  if hr_mode:
    data = pd.read_csv(rd_path+ hr_file_name +'_SV_QC.csv',header=0)
  else:
    data = pd.read_csv(rd_path+ file_name +'_SV_QC.csv',header=0)
  # Load the data from the CSV file
  if hr_mode: file_name = hr_file_name
  # To "flip" the data from row 2 to row 34 (1-based indexing in the description, so it's 1 to 33 in 0-based indexing),
  # we will reverse these rows.
  flipped_data = data.iloc[0:33].iloc[::-1]

  # To maintain the structure of the original dataframe, we'll reappend the flipped part back to any rows above and below it
  # First, slice any potential rows above and below the specified range
  rows_above = data.iloc[:0]  # Rows before the second row
  rows_below = data.iloc[33:]  # Rows after the 34th row

  # Reassemble the dataframe with the flipped part in its correct position
  flipped_full_data = pd.concat([rows_above, flipped_data, rows_below])

  # Reset the index to reflect the original ordering
  flipped_full_data.reset_index(drop=True, inplace=True)

  flipped_full_data.head()

  # The first column is the y-axis labels, and the first row is the x-axis labels.
  x_labels = flipped_full_data.columns[1:].astype(float)  # Assuming numeric labels starting from second column
  y_labels = flipped_full_data.iloc[1:, 0].astype(float)  # Assuming numeric labels starting from second row
  z_values = flipped_full_data.iloc[1:, 1:].astype(float)  # The matrix of values starting from second row and column


  # Create a meshgrid for the x and y labels
  #X, Y = np.meshgrid(x_labels, y_labels)
  #X=[0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23]   #速度 Y
  #X=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 ,3.2 ,3.6 ,4 ,4.8 ,5.6 ,6.4 ,7.8 ,8 ,9.6 ,11.2 ,12.8 ,14.4 ,16 ,19.2 ,22.4] #直徑 X
  #Y=[0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 7.8, 8, 9.6, 11.2, 12.8, 14.4, 16] #直徑 X

  X=[0,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23,26]   #直徑 X
  Y=[0,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 7.2, 8, 9.6, 11.2, 12.8, 14.4,16, 19.2,22.4] #速度 Y


  X=np.array(X)
  Y=np.array(Y)
  z_values_copy=z_values.copy()
  z_values_copy = np.asarray(z_values_copy)
  z_values_copy[z_values_copy==0] = np.nan

  # Create the plot
  fig, ax = plt.subplots(figsize=(18, 12))  # Adjust the size to match your requirements
  plt.tick_params(axis='x', labelsize=8,rotation=45)
  plt.tick_params(axis='y', labelsize=8)
  c = ax.pcolormesh(X, Y, z_values_copy, cmap='viridis',shading = 'flat')  # Create the heatmap

  # Add the function line to the plot
  aD=np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.75,3.25,3.75,4.25,4.75,5.5,6.5 ,7.5 ,8.5 ,9.5 ,11,13,15,17,19,21,24.5])
  x_line = np.linspace(x_labels.min(), x_labels.max(), 1000)  # A smooth line
  y_line = 9.65 - 10.3 * np.exp(-0.6 * x_line)
  ax.plot(x_line, y_line,color='#1f77b4', label='y=9.65-10.3*e^(-0.6*x)')  # The equation given

  QV=-0.1021+4.932*x_line-0.9551*(x_line**2)+0.07934*(x_line**3)-0.002362*(x_line**4)
  #ax.plot(x_line,QV,'k-',color='blue',zorder=3,label='ref. V$_t$')
  ax.plot(x_line,QV,'k-',color='red',zorder=3,label='-0.1021+4.932*x_line-0.9551*(x_line**2)+0.07934*(x_line**3)-0.002362*(x_line**4)')
  # Add color bar
  fig.colorbar(c)

  # Set labels and title
  ax.set_title('Number & Velocity & Diameter (Parsivel)\n'+ file_name, size=20)
  ax.set_xlabel('Diameter(mm)')
  ax.set_ylabel('Velocity(m/s)')


  plt.legend()
  #plt.xticks([0.1,0.4,0.7,1.0,1.6,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8])#室內
  plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2, 12.8, 14.4, 16, 19.2, 22.4])
  plt.xticks([0, 0.125, 0.25, 0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23, 26])
  #plt.xticks([0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23])
  #plt.yticks([0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 6.8, 7.6, 8.4, 8.8, 9.6, 11.2, 12.8, 14.4, 16, 19.2, 22.4 ]) #室外

  plt.xlim(0,7)
  plt.ylim(0,15)
  plt.grid()
  # Save the figure
  plt.savefig(rd_path + file_name + "_SV_QC.png", dpi=600)

  # Display the plot
  #plt.show()
  plt.close()