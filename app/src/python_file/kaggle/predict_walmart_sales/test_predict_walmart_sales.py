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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor 
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from python_file.kaggle.common import common_funcs as cf


DATA_DIR = "src/sample_data/Kaggle/predict_walmart_sales"
TRAIN_PATH = f'{DATA_DIR}/sales_train_evaluation.csv'
CALENDAR_PATH = f'{DATA_DIR}/calendar.csv'
PRICE_PATH = f'{DATA_DIR}/sell_prices.csv'
SAMPLE_SUBMISSION_PATH = f'{DATA_DIR}/sample_submission.csv'

# read_csvする際のdtypeを予め決定しておく
CAL_DTYPES={
    "event_name_1": "category", 
    "event_name_2": "category", 
    "event_type_1": "category", 
    "event_type_2": "category", 
    "weekday": "category", 
    'wm_yr_wk': 'int16', 
    "wday": "int16",
    "month": "int16", 
    "year": "int16", 
    "snap_CA": "float32", 
    'snap_TX': 'float32', 
    'snap_WI': 'float32'
}
PRICE_DTYPES = {
    "store_id": "category", 
    "item_id": "category", 
    "wm_yr_wk": "int16",
    "sell_price":"float32" 
}

FIRST_DAY = 1800
h = 28 
max_lags = 57
tr_last = 1913 + 28 # 学習データの最終日(evaluation file 解禁)
fday = datetime(2016,4, 25) + timedelta(days=28)# 予測を始める1日目(evaluation file 解禁によりd_1941~)


# 3つのdataframeをmergeしたものを返す
def create_df(is_train=True, nrows=None, first_day=1200): 
    prices = cf.reduce_mem_usage(pd.read_csv(PRICE_PATH, dtype=PRICE_DTYPES))
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16") # カテゴリ変数をint型に変換 ⇒ 普通にcategory_encorderでもOK
            prices[col] -= prices[col].min()

    calendar = cf.reduce_mem_usage(pd.read_csv(CALENDAR_PATH, dtype=CAL_DTYPES, parse_dates=["date"]))
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            calendar[col] = calendar[col].cat.codes.astype("int16") # カテゴリ変数をint型に変換 ⇒ 普通にcategory_encorderでもOK
            calendar[col] -= calendar[col].min()

    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id'] #  カテゴリ変数のカラム名
    dtype = {numcol: "float32" for numcol in numcols} # read_csv時の型指定(first_day~最終日まで)    
    dtype.update({catcol: "category" for catcol in catcols if catcol != "id"})
    dt = cf.reduce_mem_usage(pd.read_csv(TRAIN_PATH, nrows=nrows, usecols=[*catcols, *numcols], dtype=dtype)) # nrowは上から○○行目までreadし, usecolsは指定したカラムのみreadする
    for catcol in catcols:
        if catcol != "id":
            dt[catcol] = dt[catcol].cat.codes.astype("int16") # カテゴリ変数をint型に変換 ⇒ 普通にcategory_encorderでもOK
            dt[catcol] -= dt[catcol].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+28+1): # test用の時は予測する部分のカラム(d_1914~d_1941)を追加し、nanで埋めておく。 
            dt[f"d_{day}"] = np.nan 

    melted_dt = pd.melt(
        dt,
        id_vars=catcols, # 各日付の部分(カテゴリ変数以外の部分)を縦に並べていく
        value_vars=[col for col in dt.columns if col.startswith("d_")],
        var_name = "d",
        value_name = "sales"
    )

    merged_dt = pd.merge(
        melted_dt,
        calendar,
        how="left", 
        on="d"
    )
    merged_dt = pd.merge(
        merged_dt,
        prices,
        how="left", 
        on=["store_id", "item_id", "wm_yr_wk"]
    )
    return merged_dt


def create_feature(df):
    lags = [7, 28]
    lag_cols = [f'lag_{lag}' for lag in lags]

    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag) # idでグループ分けし、salesを7，28日ずつずらして過去のデータを特徴量として入れる
    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            # ラグ変数追加後、各ラグ変数の(7，28日前)平均値を列として追加(7、28日分の平均(win))
            df[f"rmean_{lag}_{win}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(win).mean())

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    # "date"の列(datetime型)から、月日などを取得し特徴量として加える。
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16") # 特徴量に既に存在する場合
        else:
            df[date_feat_name] = getattr(df["date"].dt, date_feat_func).astype("int16") # 存在していない場合
    return df

def main() -> None:
    if not os.path.isfile(f"{DATA_DIR}/lgb_model.pkl"):
        df = create_df(is_train=True, first_day=FIRST_DAY)
        df = create_feature(df)

        print(df)
    #     categorical_cols = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    #     useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

    #     train_x = df.drop(useless_cols, axis=1)
    #     train_y = df.loc[:, "sales"]
    #     train_cols = train_x.columns

    #     train_data = lgb.Dataset(train_x, train_y, categorical_feature=categorical_cols, free_raw_data=False) # categorical_featureを指定することでエンコーディングをしてくれる(自分でlabel_encorderしてもよい)  
        
    #     # 学習データのサブサンプルを作成(学習データの中からランダムに抽出) ⇒ サブサンプルであり実際の検証用データではない
    #     fake_valid_idx = np.random.choice(len(train_x), 1000000)
    #     fake_valid_data = lgb.Dataset(train_x.iloc[fake_valid_idx], train_y.iloc[fake_valid_idx], categorical_feature=categorical_cols, free_raw_data=False)

    #     # 学習時のパラメータ設定
    #     params = {
    #         "num_threads": 2,
    #         "objective" : "poisson",
    #         "metric" :"rmse",
    #         "force_row_wise" : True,
    #         "learning_rate" : 0.1,
    #         "sub_row" : 0.75,
    #         "bagging_freq" : 1,
    #         "lambda_l2" : 0.1,
    #         'verbosity': 1,
    #         'num_iterations' : 2500,
    #     }

    #     model = lgb.train(
    #         params, 
    #         train_data, 
    #         valid_sets=[fake_valid_data], 
    #         verbose_eval=50,
    #         early_stopping_rounds=10,
    #     )
    #     joblib.dump(model, f"{DATA_DIR}/lgb_model.pkl")
    
    # model = joblib.load(f"{DATA_DIR}/lgb_model.pkl")

    # alphas = [1.023, 1.018, 1.013]
    # weights = [1/len(alphas)]*len(alphas) # [0.33333333, 0.33333333, 0.333333333]
    # sub = 0.
    
    # for icount, (weight, alpha) in enumerate(zip(weights, alphas)):
    #     te = create_df(False) # テストデータにはd_1943~1970までのカラムが追加されている(中身はすべてNan)。
    #     cols = [f"F{i}" for i in range(1,29)]
    #     print(te)

    #     for tdelta in range(0,28):
    #         day = fday + timedelta(days=tdelta)
    #         tst = te[
    #             (te.date >= day - timedelta(days=max_lags)) & 
    #             (te.date <= day)
    #         ].copy()
    #         print(tst)
    #         _df = create_feature(tst)
    #         tst = tst.loc[tst.date == day , train_cols]
    #         te.loc[te.date == day, "sales"] = alpha * model.predict(tst) # magic multiplier by kyakovlev

    #     te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
    #     te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
    #     te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
    #     te_sub.fillna(0., inplace = True)
    #     te_sub.sort_values("id", inplace = True)
    #     te_sub.reset_index(drop=True, inplace = True)
    #     te_sub.to_csv(f"submission_{icount}.csv",index=False)
    #     if icount == 0 :
    #         sub = te_sub
    #         sub[cols] *= weight
    #     else:
    #         sub[cols] += te_sub[cols]*weight
    #     # これでevaluationの部分は完成


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executer:
        executer.submit(main()) # CPU4つ使っている。
    