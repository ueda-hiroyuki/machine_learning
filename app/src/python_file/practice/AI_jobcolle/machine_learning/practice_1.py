import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

# Pipelineの定義：一連の処理をパイプラインで表現する。 
ANY_PIPELINES = {
    'ols': Pipeline([
        ('scl',StandardScaler()), # 標準化関数
        ('est',LinearRegression()) # 線形回帰
    ]),
    'ridge1': Pipeline([
        ('scl',StandardScaler()), # 標準化関数
        ('est',Ridge(alpha=1.0)) # Ridge回帰
    ]),
    'ridge2': Pipeline([
        ('scl',StandardScaler()), # 標準化関数
        ('est',Ridge(alpha=20.0)) # Ridge回帰
    ]),
    'gbr1': Pipeline([
        ('scl',StandardScaler()), # 標準化関数
        ('est',GradientBoostingRegressor(n_estimators = 200, max_depth = 5, random_state = 0)) # 勾配ブースティング回帰
    ]),
    'gbr2': Pipeline([
        ('scl',StandardScaler()), # 標準化関数
        ('est',GradientBoostingRegressor(n_estimators = 200, max_depth = 10, random_state = 0)) # 勾配ブースティング回帰
    ])
}


def load_boston_dataset():
    dataset = load_boston()
    data = dataset.data
    data_column = dataset.feature_names
    target = dataset.target

    df_x = pd.DataFrame(data, columns=data_column)
    df_y = pd.Series(target, name='label')
    return df_x, df_y


def fit_pred():
    df_x, df_y = load_boston_dataset()

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
    df_x, df_y = load_boston_dataset()

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


def ridge_vs_ols_holdout():
    df_x, df_y = load_boston_dataset()
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=1)
    
    # 既に定義されているパイプラインを1つずつまわしていく
    for name, model in ANY_PIPELINES.items():
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        score = r2_score(test_y, y_pred)
        print(f"{name}: {score}") # scoreにあまり変化はない印象


def ridge_vs_ols_cv():
    df_x, df_y = load_boston_dataset()
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=1)
    
    # 交差検証用のKfoldの定義
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # 既に定義されているパイプラインを1つずつまわしていく
    for name, model in ANY_PIPELINES.items():
        result = cross_val_score(
            model,
            df_x,
            df_y,
            cv=kf,
            scoring='r2',
        )
        print(f"{name}: {result}")

def grid_search():
    df_x, df_y = load_boston_dataset()
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.2, random_state=1)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    steps = [
        ("scl", StandardScaler()),
        ("gbr", GradientBoostingRegressor())
    ]
    params = [
        {
            "gbr__n_estimators": [100, 200, 500],
            "gbr__max_depth": [3, 5, 10],
            "gbr__random_state": [0]
        }
    ]

    pipeline = Pipeline(steps=steps)
    grid_search_cv = GridSearchCV(pipeline, param_grid=params, cv=kf, refit=True)
    grid_search_cv.fit(train_x, train_y)
    print(f"best parem: {grid_search_cv.best_params_}")

    pred = grid_search_cv.predict(test_x)
    score = r2_score(test_y, pred)
    print(f"score: {score}")


if __name__ == "__main__":
    grid_search()