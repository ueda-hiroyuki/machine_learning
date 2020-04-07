import cv2
import os, glob
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# 学習させる画像サイズと画像のパスを設定する
image_size = (64,32)
fish_path = "../sample_data/fish_imgs/fish"
no_fish_path = "../sample_data/fish_imgs/no_fish"

x = [] # 画像データ
y = [] # ラベルデータ

# 画像データを読み込んで配列に変換
def read_imgs(path, label):
    files = glob.glob(f'{path}/*.jpg')
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, image_size)
        img_data = img.reshape(-1, ) # 1次元配列に変換
        x.append(img_data) 
        y.append(label)

read_imgs(fish_path, 1)
read_imgs(no_fish_path, 0)

# 学習用データとテスト用データを分割する。
# 画像データたちを1次元配列のリストとして学習させる
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 画像データを学習させる
clf = RandomForestClassifier().fit(x_train, y_train)

# 精度の確認
y_pred = clf.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(y_test)
print(y_pred)
print(f'スコアは {score} です！')

# 学習モデルを保存
joblib.dump(clf, "../sample_data/fish_model.pkl")
