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
  PATH_ROOT     = "D:/"
elif ( sys.platform == "win32" and getpass.getuser() == "Jason") :
  PATH_ROOT     = "D:/"
  #elif ( sys.platform == "win32" and getpass.getuser() == "rain") :
#  PATH_ROOT     = "H:/PC_ubuntu_space/"

PROJECT_PATH  = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/"

def rainfall_measurement_path_def():
    RAIN_DATA_SEL = "RR-20231206-outdoor/"
    RESULT_ID     = "1206-outdoor"
    KEY_CHAR      = "R"   #add

             #底線後名稱
    
    #============>READ_PATH = "PATH_TO_DATA"
    READ_DATA_PATH       = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "ori_grab_img/"
    WRITE_R_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_R/"
    WRITE_D_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_D/"
    WRITE_THERSHOLD_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_img/"

    #============>MODEL_READ_PATH = "PATH_TO_TRAINED_MODEL"
    MODEL_READ_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/model/"
    
    #============>WRITE_PATH = "PATH_TO_WRITE"
    RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
    RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
    RESULT_PATH_ARRIMG  = RESULT_PATH + "ARR_ans_" + RESULT_ID + "/"
    RESULT_PATH_MODIFY  = RESULT_PATH + "Modify_" + RESULT_ID + "/"

    return RESULT_ID, WRITE_R_PATH, MODEL_READ_PATH, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_ARRIMG, RESULT_PATH_MODIFY, KEY_CHAR

def draw_path_def():
  RESULT_ID     = "20231206-outdoor"            #底線後名稱

  RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
  RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
  RESULT_PATH_ARRIMG  = RESULT_PATH + "ARR_ans_" + RESULT_ID + "/"
  return RESULT_ID, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_ARRIMG


def detect_raindrop_path_def():
  RAIN_DATA_SEL = "RR-20231206-outdoor/"

  READ_DATA_PATH       = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "ori_grab_img/"
  WRITE_R_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_R/"
  WRITE_D_PATH         = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "algorithm1_out_D/"
  WRITE_THERSHOLD_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/rainData/" + RAIN_DATA_SEL + "threshold_img/"

  return READ_DATA_PATH, WRITE_R_PATH, WRITE_D_PATH, WRITE_THERSHOLD_PATH

def execl_modify_path_def():
    RESULT_ID     = "20231206-outdoor"            #底線後名稱

    RESULT_PATH = PATH_ROOT + "Raindrop_folder/Rainfall_project_2023/result/result_" + RESULT_ID + "/"
    RESULT_PATH_ANALYZE = RESULT_PATH + "analyze_" + RESULT_ID + "/"
    RESULT_PATH_MODIFY  = RESULT_PATH + "Modify_" + RESULT_ID + "/"

    return RESULT_ID, RESULT_PATH, RESULT_PATH_ANALYZE, RESULT_PATH_MODIFY
