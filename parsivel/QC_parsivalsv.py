import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

filename='Parsivel_20240501_SV_QC'
csv = '.csv'
png = '.png'
new_file_path = 'F:/Raindrop_folder/Rainfall_project_2023/Parsivel_QC_data/'+ filename + csv


# Reload the data with proper headers
data = pd.read_csv(new_file_path, header=0)
# Load the data from the CSV file

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
plt.tick_params(axis='x', labelsize=20,rotation=45)
plt.tick_params(axis='y', labelsize=20)
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
ax.set_title('Number & Velocity & Diameter (Parsivel)\n'+ filename, size=20)
ax.set_xlabel('Diameter(mm)')
ax.set_ylabel('Velocity(m/s)')


plt.legend()
#plt.xticks([0.1,0.4,0.7,1.0,1.6,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8])#室內
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2, 12.8, 14.4, 16, 19.2, 22.4])
# plt.xticks([0, 0.125, 0.25, 0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23, 26])
#plt.xticks([0 ,0.125 ,0.25 ,0.375 ,0.5 ,0.625 ,0.75 ,0.875 ,1 ,1.125 ,1.25 ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10 ,12 ,14 ,16 ,18 ,20 ,23])
#plt.yticks([0 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,1 ,1.2 ,1.4 ,1.6 ,1.8 ,2 ,2.4 ,2.8 , 3, 3.2, 3.6, 4, 4.8, 5.6 ,6.4, 6.8, 7.6, 8.4, 8.8, 9.6, 11.2, 12.8, 14.4, 16, 19.2, 22.4 ]) #室外
plt.yticks([0.1,  0.5, 1, 1.5 , 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8, 5.6, 6.4, 7.2, 8, 9.6, 11.2,14.4])
plt.xticks([0,0.125,0.25,0.5 ,0.75  ,1,1.25  ,1.5 ,1.75 ,2 ,2.25 ,2.5 ,3 ,3.5 ,4 ,4.5 ,5 ,6 ,7 ,8 ,9 ,10])
plt.xlim(0,5)
plt.ylim(0,12)
plt.grid()
# Save the figure
plt.savefig(filename + ".png", dpi=600)

# Display the plot
plt.show()