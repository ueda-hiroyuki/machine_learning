import matplotlib.pyplot as plt
import pandas as pd
import typing as t
from datetime import datetime 

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby([df.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    df = df[['date','item_id','shop_id','item_cnt_day']]
    df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index() 
    df = df.drop(["2015-11", "2015-12"], axis=1)    
    return df

def remove_outlier(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['item_cnt_day'] < 1000) & (df['item_price'] < 100000)]
    return df

def main():
    ## parse_dates: 文字列で表示されていた日時をdatetime型として読み込む
    train = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/sales_train.csv", parse_dates=["date"], infer_datetime_format=True)
    # test = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/test.csv")
    # items = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/items.csv")
    # item_categories = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/item_categories.csv")
    print(train.head(5))
    train = remove_outlier(train)
    train = train.drop(["date_block_num","item_price"], axis=1)
    train_df = preprocess(train)
    print(train_df.head(20))


if __name__ == "__main__":
    # 2015年11月の売り上げを予測する。
    main()