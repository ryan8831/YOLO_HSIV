import os
import shutil
import getpass
import sys
import string

if (sys.platform == "darwin") :
  PATH_ROOT    = "/Users/matthew/project/"
elif ( sys.platform == "win32" and getpass.getuser() == "s9303") :   # matthew's 桌機
  PATH_ROOT     = "T:/"
elif ( sys.platform == "win32" and getpass.getuser() == "rain") :    # 雨滴普儀 共用桌機
  PATH_ROOT     = "H:/PC_ubuntu_space/"
elif ( sys.platform == "win32" and getpass.getuser() == "aesnr123") :  # Ryan's Lab PC
  PATH_ROOT     = "F:/"
elif ( sys.platform == "win32" and getpass.getuser() == "Jason") :
  PATH_ROOT     = "D:/"
  #elif ( sys.platform == "win32" and getpass.getuser() == "rain") :
#  PATH_ROOT     = "H:/PC_ubuntu_space/"



def move_images_to_folders(fhead,base_path):
    # 讀取所有檔案和資料夾
    entries = os.listdir(base_path)
    folders = { e for e in entries if os.path.isdir(os.path.join(base_path, e))}
    files = [f for f in entries if os.path.isfile(os.path.join(base_path, f)) and f.endswith('.png')]

    for file in files:
        # 提取檔名中的小時編號（檔名結構是固定的）
        folder_suffix = file[11:13]  # 第12和13個字元
        target_folder_name = f"{fhead}{folder_suffix}"

        # 檢查是否存在對應的子資料夾，如果不存在，則建立
        if target_folder_name not in folders:
            os.mkdir(os.path.join(base_path, target_folder_name))
            folders.add(target_folder_name)

        # 移動檔案到對應的子資料夾
        shutil.move(os.path.join(base_path, file), os.path.join(base_path, target_folder_name, file))




# =======================================================================  Main ======================================================================= #

F_HEAD         = "o"  # RR or TT
DATA_SEL       = "RR-20170310/"

F_HEAD_FOLDER  = "algorithm1_out_R/"      if (F_HEAD == "RR") else "threshold_img/"


#FOLDER = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + DATA_SEL +  F_HEAD_FOLDER
FOLDER = 'F:/RR-20240501/ori_grab_img/'
move_images_to_folders( F_HEAD, FOLDER)
