# python_code_source
 AI part code
- - -
## 目錄
   * 資料放在raindata
    * raindata裡的資料名稱為RR-OOOOXXxx裡面包含
        1. ori_grab_img -> 原圖
        2. algorithm1_out_D ->背景圖片
        3. algorithm1_out_R ->二值化圖有做find contour
        4. threshold_img ->二值化圖沒做find contour
        5. threshold_ori -> 挑出有雨滴的原圖執行yolo所需
## main.py
* USEYOLO選擇要不要用YOLOV9
## rainfall_measurement.py
* 64行 選擇RF模型要用的特徵
    * features[['Area_Difference','Euclidean_Distance','a1','b1','a2','b2']] 
* 182行 SIGMOID閥值用來決定配對的標準
* 231行 YOLOv9det用來選擇YOLOv9要用的模型
* 253行 have_rain=any(l_diameters >=3) or any(f_diameters>=3) 3代表影像裡有直徑3公分以上的雨滴才產生ARR二值化圖
## model_train.py
1.第14、16行要更改路徑->選擇輸入輸出
- - -
# ANN_RF資料夾
## modeltest.py
* 19跟20行用來選擇要測試的資料集
* 122行test_ann代表只測試ann
* 123行plus_tree=1時代表測試ANN+RF(test_ann要=0)
* test_ann=0 plus_tree=0代表只測試RF
* 124 choice_class=11 代表選擇直徑類別多少以上的雨滴是使用RF
## randon forester.py
* 58行 test_path 選擇要拿來測試的資料集
* 59行 train_path 選擇要訓練的資料集
* 64行 savepath選擇訓練的模型存放地
* 26和28行 選擇要用那些特徵訓練 26和28行要一樣
* 34行 n_estimators決定要用多少個決策數構築成隨機森林模型