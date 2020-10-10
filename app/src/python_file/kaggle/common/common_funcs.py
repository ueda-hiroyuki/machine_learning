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
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:  # columns毎に処理
        col_type = df[col].dtypes
        if (
            col_type in numerics
        ):  # numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction_per = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print(f"Mem. usage decreased to {end_mem} Mb ({reduction_per}% reduction)")
    return df


def check_corr(df: pd.DataFrame, f_name: str) -> None:
    corr = df.corr().round(2)
    plt.figure(figsize=(50, 50))
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1)
    plt.savefig(f"src/sample_data/Kaggle/{f_name}/corr_heatmap.png")


def check_fig(df: pd.DataFrame, f_name: str) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f"src/sample_data/Kaggle/kaggle_dataset/{f_name}/{name}.png")


def check_hist(df: pd.DataFrame, f_name: str):
    for col_name, item in df.iteritems():
        plt.figure()
        df[col_name].value_counts().plot(kind="bar")
        plt.savefig(f"src/sample_data/Kaggle/{f_name}/{col_name}_hist.png")


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


def remove_outlier(df: pd.DataFrame, sigma: int):
    for i in range(len(df.columns)):
        # 列を抽出する
        col = df.iloc[:, i]

        # 平均と標準偏差
        average = np.mean(col)
        sd = np.std(col)

        # 外れ値の基準点
        outlier_min = average - (sd) * sigma
        outlier_max = average + (sd) * sigma

        # 範囲から外れている値を除く
        col[col < outlier_min] = None
        col[col > outlier_max] = None

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


def corr_column(df, threshold):

    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    # 対角線の値を0にする
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None

            # その他との相関の絶対値が大きい方を除去
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column

            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)

    return df.loc[:, df_corr.columns]
