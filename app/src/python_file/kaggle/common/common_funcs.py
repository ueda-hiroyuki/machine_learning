import re
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction_per = 100 * (start_mem - end_mem) / start_mem
    if verbose: print(f'Mem. usage decreased to {end_mem} Mb ({reduction_per}% reduction)')
    return df


def check_corr(df: pd.DataFrame, f_name: str) -> None:
    corr = df.corr().round(2)
    plt.figure(figsize=(50,50))
    sns.heatmap(corr, square=True, annot=True)
    plt.savefig(f'src/sample_data/Kaggle/{f_name}/corr_heatmap.png')


def check_fig(df: pd.DataFrame, f_name: str) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f'src/sample_data/Kaggle/kaggle_dataset/{f_name}/{name}.png')


def label_encorder(df: pd.DataFrame) -> pd.DataFrame:
    for col_name, col in df.iteritems():
        col = col.fillna(col.mode()[0])
        if col.dtypes == "object":
            le = LabelEncoder()
            le.fit(col)
            df[col_name] = le.transform(col)
        else:
            df[col_name] = col
    return df


    for col in cols:
        series = df[col]
        le = LabelEncoder()
        df[col] = le.fit_transform(series)
    return df


def remove_outlier(df: pd.DataFrame, sigma: int) -> pd.DataFrame:
    for name in df:
        series = df[name]
        z = stats.zscore(series) < sigma
        df[name] = column[z]
    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df)
    scaler_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaler_df

def minmaxscaler(df: pd.DataFrame) -> pd.DataFrame:
    mm = MinMaxScaler()
    mm.fit(df)
    mm_df = pd.DataFrame(mm.transform(df), columns=df.columns)
    return mm_df
