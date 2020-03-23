import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


TEST_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/test_data.csv'
TRAIN_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/train_data.csv'
PRICE_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/published_land_price.csv'
SIGMA = 3


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


def preprocess_train(train: pd.DataFrame, sigma: float) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train = remove_extra_str(train.drop(["都道府県名","市区町村名"], axis=1)).dropna()
    print(train)
    train_y = train.loc[:"y"]
    train_x = train.drop("y", axis=1)
    for column in train_x.columns:
        series = train_x[column]
        # uniques = list(series.value_counts().index)
        # print(type(uniques[0]))
        # if type(uniques[0]) == str:
        #     for name in uniques:
        #         print(name)
        
        
    return train_x, train_y



def preprocess_test(test: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    test = test.drop(["都道府県名","市区町村名"], axis=1)
    
    return test, test


def remove_outlier(df: pd.DataFrame, sigma: int) -> pd.DataFrame:
    for column in df:
        series = df[column]
        z = stats.zscore(series) < sigma
        df[column] = series[z]
    return df       


def remove_extra_str(df: pd.DataFrame) -> pd.DataFrame:
    re = df.apply(lambda x: x.str.contains('以上|未満|?|戦前', na=False))
    df = df.mask(re)
    return df


def main() -> None:
    train = reduce_mem_usage(pd.read_csv(TRAIN_CSV))
    test = reduce_mem_usage(pd.read_csv(TEST_CSV))
    price = reduce_mem_usage(pd.read_csv(PRICE_CSV))

    train = train.head(100000)

    train_x, train_y = preprocess_train(train, SIGMA)



if __name__ == "__main__":
    main()