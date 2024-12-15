import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
# 加載數據
def load_data(train_path, test_path):
    df_train = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)
    return df_train, df_test

def prepare_data(df_train, df_test):
    df_train['Area_Difference']=(df_train['Area2'] - df_train['Area1']).abs()
    df_train['Euclidean_Distance'] = np.sqrt((df_train['X2'] - df_train['X1'])**2 + (df_train['Y2'] - df_train['Y1'])**2)
    df_test['Area_Difference']=(df_test['Area2'] - df_test['Area1']).abs()
    df_test['Euclidean_Distance'] = np.sqrt((df_test['X2'] - df_test['X1'])**2 + (df_test['Y2'] - df_test['Y1'])**2)
    #df_test['ax_ratio']=
    X_train = df_train[['Euclidean_Distance','Area_Difference','a1','b1','a2','b2']]
    y_train = df_train['Match']
    X_test = df_test[['Euclidean_Distance','Area_Difference','a1','b1','a2','b2']]
    y_test = df_test['Match']
    return X_train, y_train, X_test, y_test

# 訓練模型
def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


# 預測與評估
def evaluate_model(rf_classifier, X_test, y_test):
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    return accuracy, report

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.title('Confusion matrix')
    plt.show()
# 主函數，整合上述步驟
def main():
    INPUT_PATH = 'F:/Raindrop_folder/Rainfall_project_2023/'
    test_path = INPUT_PATH +'rain_match_big.xlsx'
    train_path = INPUT_PATH+'rain_match.xlsx'
    df_train, df_test = load_data(train_path, test_path)
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    rf_classifier = train_model(X_train, y_train)
    accuracy, report = evaluate_model(rf_classifier, X_test, y_test)
    savepath='C:/Raindrop_folder/Rainfall_project_2023/model/v2/forseter_distance_axratio/'
    dump(rf_classifier, savepath+'random_forest_model.joblib')
    print(f'模型準確率: {accuracy}')
    print('特徵重要程度: ',rf_classifier.feature_importances_)
    print('分類報告:')
    print(report)
    

if __name__ == "__main__":
    main()