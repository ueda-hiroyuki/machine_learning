import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from kaggle.common import common_funcs as cf

TRAIN_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sales_train_validation.csv'
CALENDAR_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/calendar.csv'
PRICE_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sell_prices.csv'
SAVE_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/submission.csv'

train_meta = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

def extract_data(df: pd.DataFrame, days: int) -> pd.DataFrame:
    print(df)
    extracted_df = df.iloc[:, -days:-1]
    return extracted_df

def train_preprocess(train: pd.DataFrame, calendar: pd.DataFrame, price: pd.DataFrame, cols: t.Sequence[str]) -> pd.DataFrame:
    melted_train = dd.melt(
        train, 
        id_vars=cols,
        var_name='day',
        value_name='count'
    )
    print(melted_train.columns)
    # print(melted_train.compute().head(50))
    # print(calendar["d"].head(100), calendar.columns)
    # print(price, price.columns)
    merged_train = dd.merge(melted_train, calendar, left_on='day', right_on='d')
    print(merged_train.columns)
    print(merged_train.compute()["day"],merged_train.compute()["d"])
    # print(merged_train.head(50))
    # return merged_train

def melt(df: pd.DataFrame) -> pd.DataFrame:
    ...

def main(train_path: str, calendar_path: str, price_path: str, save_path: str) -> None:
    train = dd.read_csv(train_path)
    calendar = dd.read_csv(calendar_path)
    price = dd.read_csv(price_path)

    # _train = train.head(20)
    # _calendar = calendar.head(20)
    # _price = price.head(20)

    # print(_train)
    # print(_calendar)
    # print(_price)

    meta = train.loc[:,train_meta]
    train = train.drop(train_meta, axis=1)
    extract_train = extract_data(train, 10)
    print(type(extract_train))
    print(type(meta))

    train = dd.merge(meta, extract_train)
    train = train_preprocess(train, calendar, price, train_meta)


    


if __name__ == "__main__":
    main(TRAIN_PATH, CALENDAR_PATH, PRICE_PATH, SAVE_PATH)