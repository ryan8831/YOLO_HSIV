import getpass
import sys

# >>>>>>>>>>>>>>>> system info >>>>>>>>>>>>>>>> #
username = getpass.getuser()
os_platform   = sys.platform
# <<<<<<<<<<<<<<<<< system info <<<<<<<<<<<<<<<<< #


# >>>>>>>>>>>>>>>> path config >>>>>>>>>>>>>>>> #
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

PROJECT_PATH  = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/"
#PATH_ROOT_2   = 

def detect_raindrop_path_def():
  RAIN_DATA_SEL = "RR-20240614/"

  READ_DATA_PATH       = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "ori_grab_img/"
  WRITE_R_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_R/"
  WRITE_D_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_D/"
  WRITE_THERSHOLD_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_img/"
  WRITE_THERSHOLD_ORI_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_ori/"
  return READ_DATA_PATH, WRITE_R_PATH, WRITE_D_PATH, WRITE_THERSHOLD_PATH,WRITE_THERSHOLD_ORI_PATH


def rainfall_measurement_path_def():
    RAIN_DATA_SEL  = "RR-20240501/"  #20240331-outdoor/  
    RESULT_ID      = "RR-20240501-ALLDAY"
    KEY_CHAR       = "T"                              #  "T"  or  "R" or "o"
    SAVE_ARR_IMG   = 0                               #  要不要儲存連線圖 (方便debug)
    SHADOW_FIX     = 1                                #  要不要作殘影修正
    QC_OPERATION   = 0                                #  要不要作QC
    MODEL_Version  ="v1/" #ANN
    FORESTER_MODEL_SELECT='forseter_area_distance_axis/'  #forseter_area_distance_axis
    #============>READ_PATH = "PATH_TO_DATA"
    READ_DATA_PATH       = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_ori/"
    #WRITE_R_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_R/RR"+HR_SEL+"/"
    WRITE_R_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_R/"
    WRITE_D_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_D/"
    #WRITE_THERSHOLD_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_img/TT"+HR_SEL+"/"
    WRITE_THERSHOLD_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_img/"

    #============>MODEL_READ_PATH = "PATH_TO_TRAINED_MODEL"
    MODEL_READ_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/model/"+MODEL_Version
    FORESTER_MODEL_PATH=PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/model/"+FORESTER_MODEL_SELECT + 'random_forest_model.joblib'
    #============>WRITE_PATH = "PATH_TO_WRITE"
    RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
    RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
    RESULT_PATH_ARRIMG  = RESULT_PATH + "ARR_ans_" + RESULT_ID + "/"
    RESULT_PATH_MODIFY  = RESULT_PATH + "Modify_" + RESULT_ID + "/"

    if ( KEY_CHAR  == "T") :
      WRITE_SEL = WRITE_THERSHOLD_PATH 
    elif ( KEY_CHAR  == "R") :
      WRITE_SEL = WRITE_R_PATH
    elif ( KEY_CHAR  == "o") :
      WRITE_SEL = READ_DATA_PATH 

    return RESULT_ID, WRITE_SEL, MODEL_READ_PATH, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_ARRIMG, RESULT_PATH_MODIFY, KEY_CHAR, SAVE_ARR_IMG, SHADOW_FIX, QC_OPERATION,FORESTER_MODEL_PATH

def draw_path_def():
  RESULT_ID     = "RR-20240501-ALLDAY-YOLOV9-SHADOW-QC"            #底線後名稱
  QC_OPERATION  = 1

  RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
  RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
  RESULT_PATH_ARRIMG  = RESULT_PATH + "ARR_ans_" + RESULT_ID + "/"
  return RESULT_ID, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_ARRIMG, QC_OPERATION

def execl_modify_path_def():
    RESULT_ID     = "20240331_hr18_th5"            #底線後名稱

    RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
    RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
    RESULT_PATH_MODIFY  = RESULT_PATH + "Modify_" + RESULT_ID + "/"

    return RESULT_ID, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_MODIFY