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


def main():
    sales = pd.read_csv(TRAIN_EVALUATION_PATH)
    sales.name = 'sales' # dataframeの名前を指定している
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar.name = 'calendar'
    prices = pd.read_csv(PRICE_PATH)
    prices.name = 'prices'
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
    merged_dataset["revenue"] = merged_dataset["sell_price"] * merged_dataset["sales"] 

    print(merged_dataset.loc[:, [*FIXED_COLS, "sell_price"]])
    group_price_store = merged_dataset.groupby(['state_id','store_id','item_id'],as_index=False)['sell_price'].mean().dropna() # 各item_idの平均の売り上げ

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_dataset.columns if merged_dataset[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_dataset) 

    print(encorded_data)

    # ラグ変数を特徴量として加える。
    lags = [1,2,3,6,12,24,36]
    for lag in lags:
        # lagsで指定した日数分前の傾向データを加える
        encorded_data[f"sales_lag_{lag}"] = encorded_data.groupby(FIXED_COLS)["sales"].shift(lag).astype(np.float16)

    # target mean encoding (transform: groupbyしても行は圧縮されない)
    encorded_data['iteam_sales_avg'] = encorded_data.groupby('item_id')['sales'].transform('mean').astype(np.float16)
    encorded_data['state_sales_avg'] = encorded_data.groupby('state_id')['sales'].transform('mean').astype(np.float16)
    encorded_data['store_sales_avg'] = encorded_data.groupby('store_id')['sales'].transform('mean').astype(np.float16)
    encorded_data['cat_sales_avg'] = encorded_data.groupby('cat_id')['sales'].transform('mean').astype(np.float16)
    encorded_data['dept_sales_avg'] = encorded_data.groupby('dept_id')['sales'].transform('mean').astype(np.float16)
    encorded_data['cat_dept_sales_avg'] = encorded_data.groupby(['cat_id','dept_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['store_item_sales_avg'] = encorded_data.groupby(['store_id','item_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['cat_item_sales_avg'] = encorded_data.groupby(['cat_id','item_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['dept_item_sales_avg'] = encorded_data.groupby(['dept_id','item_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['state_store_sales_avg'] = encorded_data.groupby(['state_id','store_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['state_store_cat_sales_avg'] = encorded_data.groupby(['state_id','store_id','cat_id'])['sales'].transform('mean').astype(np.float16)
    encorded_data['store_cat_dept_sales_avg'] = encorded_data.groupby(['store_id','cat_id','dept_id'])['sales'].transform('mean').astype(np.float16)



if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executer:
        executer.submit(main()) # CPU2つ使っている。
    