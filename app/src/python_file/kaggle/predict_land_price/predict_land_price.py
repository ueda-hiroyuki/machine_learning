import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


TEST_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/test_data.csv'
TRAIN_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/train_data.csv'
PRICE_CSV = 'src/sample_data/Kaggle/kaggle_dataset/predict_land_price/published_land_price.csv'


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


def preprocess_train(train: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.drop(["都道府県名","市区町村名"], axis=1)
    train_y = train.loc[:"y"]
    train_x = train.drop("y", axis=1)
    for column in train_x.columns:
        series = train_x[column]
        uniques = list(series.value_counts().index)
        print(type(uniques[0]))
        if type(uniques[0]) == str:
            for name in uniques:
                print(name)
        
        
    return train_x, train_y



def preprocess_test(test: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    test = test.drop(["都道府県名","市区町村名"], axis=1)
    
    return test, test


def main() -> None:
    train = reduce_mem_usage(pd.read_csv(TRAIN_CSV))
    test = reduce_mem_usage(pd.read_csv(TEST_CSV))
    price = reduce_mem_usage(pd.read_csv(PRICE_CSV))

    train_x, train_y = preprocess_train(train)



if __name__ == "__main__":
    main()