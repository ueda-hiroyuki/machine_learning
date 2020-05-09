import numpy as np
import pandas as pd
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split



def fit_pred():
    dataset = load_boston()
    data = dataset.data
    data_column = dataset.feature_names
    target = dataset.target

    df_x = pd.DataFrame(data, columns=data_column)
    df_y = pd.Series(target, name='label')

    # define model
    est = GradientBoostingRegressor(
        n_estimators = 100, # 分類器の数(ブースティング数)
        max_depth = 5, # 決定木の深さ
        random_state = 0 # intで設定すると毎回同じ値となる
    )

    est.fit(df_x, df_y) # 学習用データと学習用ラベルの学習を行う
    y_pred = est.predict(df_x) # あるデータに対して、予測値を算出(今回は学習データをそのまま使用)

    score_r2 = r2_score(target, y_pred) # 決定係数(実値と予測値の相関をみる)
    score_rmse = mean_squared_error(target, y_pred, squared=False) # 二乗平均平方根誤差
    print(score_r2, score_rmse)

def holdout():
    dataset = load_boston()
    data = dataset.data
    data_column = dataset.feature_names
    target = dataset.target

    df_x = pd.DataFrame(data, columns=data_column)
    df_y = pd.Series(target, name='label')
    
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=1)

    # 二つのモデルを構築(比較用)
    # Standard model
    est1 = GradientBoostingRegressor(max_depth=3,random_state=1)
    est1.fit(train_x, train_y)
    # Complex model
    est2 = GradientBoostingRegressor(max_depth=10,random_state=1) 
    est2.fit(train_x, train_y)

    # モデルパフォーマンス指標(R2)の取得
    # for training data
    r2_est1_train = r2_score(train_y, est1.predict(train_x))
    r2_est2_train = r2_score(train_y, est2.predict(train_x))
    # for test data
    r2_est1_test = r2_score(test_y, est1.predict(test_x))
    r2_est2_test = r2_score(test_y, est2.predict(test_x))

    # 決定木が深くなりすぎると過学習を起こしがち(適当なところを選択するべき)
    print(f"model1 ⇒ train: {r2_est1_train}, test: {r2_est1_test}")
    print(f"model2 ⇒ train: {r2_est2_train}, test: {r2_est2_test}")



if __name__ == "__main__":
    holdout()