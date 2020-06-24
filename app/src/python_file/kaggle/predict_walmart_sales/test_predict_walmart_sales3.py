import gc
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor 
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from python_file.kaggle.common import common_funcs as cf


DATA_DIR = "src/sample_data/Kaggle/predict_walmart_sales"
TRAIN_EVALUATION_PATH = f'{DATA_DIR}/sales_train_evaluation.csv'
TRAIN_VALIDATION_PATH = f'{DATA_DIR}/sales_train_validation.csv'
CALENDAR_PATH = f'{DATA_DIR}/calendar.csv'
PRICE_PATH = f'{DATA_DIR}/sell_prices.csv'
SAMPLE_SUBMISSION_PATH = f'{DATA_DIR}/sample_submission.csv'

FIXED_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

"""
d_1914~d_1941までの28日間の各アイテムの売り上げを予測する
"""

def get_best_params(train_x: t.Any, train_y: t.Any, valid_x: t.Any, valid_y: t.Any) -> t.Any:
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
    best_params = {}
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt', 
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=10000,
        early_stopping_rounds=20,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history
    )
    return best_params

def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=10000,
    )
    return model

def main():
    sales = pd.read_csv(TRAIN_EVALUATION_PATH)
    sales.name = 'sales' # dataframeの名前を指定している
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar.name = 'calendar'
    prices = pd.read_csv(PRICE_PATH)
    prices.name = 'prices'
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    sample_submission.name = 'submission'
    ids = sample_submission.loc[:,"id"]
    if not os.path.isfile(f"{DATA_DIR}/dataset.pkl"):
        for col in [f"d_{i}" for i in range(1942, 1970)]:
            sales[col] = 0
            sales[col] = sales[col].astype(np.int16)

        melted_sales = pd.melt( # wide ⇒ long フォーマット変換
            sales,
            id_vars = FIXED_COLS,
            var_name="d", 
            value_name='sales', 
        )
        merged_dataset = pd.merge(
            melted_sales,
            calendar,
            how="left",
            on=["d"]
        )
        merged_dataset = pd.merge(
            merged_dataset,
            prices,
            how="left",
            on=["store_id", "item_id", "wm_yr_wk"]
        ) # 3つのdataframeを結合
        merged_dataset["revenue"] = merged_dataset["sell_price"] * merged_dataset["sales"] # 総売上の特徴量を追加
        group_price_store = merged_dataset.groupby(['state_id','store_id','item_id'],as_index=False)['sell_price'].mean().dropna() # 各item_idの平均の売り上げ

        # category_encodersによってカテゴリ変数をencordingする
        categorical_columns = [c for c in merged_dataset.columns if merged_dataset[c].dtype == 'object']
        ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
        encorded_data = ce_oe.fit_transform(merged_dataset) 

        # ラグ変数を特徴量として加える。
        lags = [1,2,3,6,12,24,36]
        for lag in lags:
            # lagsで指定した日数分前の傾向データを加える
            encorded_data[f"sales_lag_{lag}"] = encorded_data.groupby(FIXED_COLS)["sales"].shift(lag).astype(np.float16)

        # target encoding (transform: groupbyしても行は圧縮されない) ⇒ 目的変数からカテゴリ変数を数値変換する手法
        encorded_data['iteam_sales_avg'] = encorded_data.groupby('item_id')['sales'].transform('mean').astype(np.float16) # 各item_idの平均売り上げ数
        encorded_data['state_sales_avg'] = encorded_data.groupby('state_id')['sales'].transform('mean').astype(np.float16) # 各state_idの平均売り上げ数
        encorded_data['store_sales_avg'] = encorded_data.groupby('store_id')['sales'].transform('mean').astype(np.float16) # 各store_idの平均売り上げ数
        encorded_data['cat_sales_avg'] = encorded_data.groupby('cat_id')['sales'].transform('mean').astype(np.float16) # 各cat_idの平均売り上げ数
        encorded_data['dept_sales_avg'] = encorded_data.groupby('dept_id')['sales'].transform('mean').astype(np.float16) # 各dept_idの平均売り上げ数
        encorded_data['cat_dept_sales_avg'] = encorded_data.groupby(['cat_id','dept_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['store_item_sales_avg'] = encorded_data.groupby(['store_id','item_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['cat_item_sales_avg'] = encorded_data.groupby(['cat_id','item_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['dept_item_sales_avg'] = encorded_data.groupby(['dept_id','item_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['state_store_sales_avg'] = encorded_data.groupby(['state_id','store_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['state_store_cat_sales_avg'] = encorded_data.groupby(['state_id','store_id','cat_id'])['sales'].transform('mean').astype(np.float16)
        encorded_data['store_cat_dept_sales_avg'] = encorded_data.groupby(['store_id','cat_id','dept_id'])['sales'].transform('mean').astype(np.float16)

        encorded_data["rolling_sales_mean"] = encorded_data.groupby(FIXED_COLS)["sales"].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16) # グループの直近7日間の売り上げ平均を特徴量として加える。
        encorded_data['expanding_sold_mean'] = encorded_data.groupby(FIXED_COLS)['sales'].transform(lambda x: x.expanding(2).mean()).astype(np.float16) # グループの累積平均を特徴量として加える。

        encorded_data = encorded_data[encorded_data["d"] >= 36] # ラグ変数のカラムは36日以前までのデータがNaNになっている為、その部分は切り落とす
        joblib.dump(encorded_data, f"{DATA_DIR}/dataset.pkl", compress=3)
    
    dataset = joblib.load(f"{DATA_DIR}/dataset.pkl")
    # d_36 ~ d_1970 のうち1914~1941の28日間を評価用データ(validation)、1942~1969の28日間を検証用データ(evaluation)に設定する
    valid = dataset[(dataset["d"] >= 1914) & (dataset["d"] < 1942)].loc[:, ["id", "d", "sales"]]
    test = dataset[(dataset["d"] >= 1942)].loc[:, ["id", "d", "sales"]]
    valid_preds = valid.loc[:, "sales"]
    eval_preds = test.loc[:, "sales"]

    stores = list(dataset["store_id"].unique())
    for store in stores:
        _df = dataset[dataset["store_id"] == store]
        train_x = _df[_df["d"] < 1914].drop("sales", axis=1)
        train_y = _df[_df["d"] < 1914].loc[:, "sales"]
        valid_x = _df[(_df["d"] >= 1914) & (_df["d"] < 1942)].drop("sales", axis=1)
        valid_y = _df[(_df["d"] >= 1914) & (_df["d"] < 1942)].loc[:, "sales"]
        test_x = _df[_df["d"] >= 1942].drop("sales", axis=1)

        if not os.path.isfile(f"{DATA_DIR}/lgb_model_{store}.pkl"):
            best_params = get_best_params(train_x, train_y, valid_x, valid_y)
            _lgb_train = lgb.Dataset(train_x, train_y)
            _lgb_valid = lgb.Dataset(valid_x, valid_y, reference=_lgb_train)
            lgb_model = get_model(_lgb_train, _lgb_valid, best_params)
            print(f"START FIT to store_id: {store} !!")
            
            joblib.dump(lgb_model, f"{DATA_DIR}/lgb_model_{store}.pkl")
        else:
            lgb_model = joblib.load(f"{DATA_DIR}/lgb_model_{store}.pkl")
        valid_preds[valid_x.index] = lgb_model.predict(valid_x) # 選択されたstore_id、日にちのindex部分に予測結果を当てはめていく。
        eval_preds[test_x.index] = lgb_model.predict(test_x)

    valid["sales"] = valid_preds
    validation = valid.loc[:, ["id", "d", "sales"]]
    validation = validation.pivot(index='id', columns='d', values='sales').reset_index(drop=True)
    validation.columns = [f"F{i}" for i in range(1, 29)]

    test["sales"] = eval_preds
    evalution = test.loc[:, ["id", "d", "sales"]]
    evalution = evalution.pivot(index='id', columns='d', values='sales').reset_index(drop=True)
    evalution.columns = [f"F{i}" for i in range(1, 29)]
    
    submission = pd.concat([validation, evalution], axis=0).reset_index(drop=True)
    submission = submission.where(submission>0, 0)
    submission = pd.concat([ids, submission], axis=1)
    print(submission)

    submission.to_csv(f"{DATA_DIR}/submission_4.csv", index=False)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executer:
        executer.submit(main()) # CPU2つ使っている。
    