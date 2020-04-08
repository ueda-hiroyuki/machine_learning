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
SAMPLE_SUBMISSION_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sample_submission.csv'
SAVE_DIR = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales'

TRAIN_META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
ADD_COLS = ["day", "date", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"] 

def extract_data(
    df: pd.DataFrame, 
    days: int
) -> pd.DataFrame:
    extracted_df = df.iloc[:, -days:]
    return extracted_df

def train_preprocess(
    train: pd.DataFrame, 
    cols: t.Sequence[str], 
) -> pd.DataFrame:
    train = pd.melt(
        train, 
        id_vars=cols,
        var_name='day',
        value_name='count'
    )
    train["use"] = "train"
    return train


def test_preprocess(submission_df: pd.DataFrame, meta_df: pd.DataFrame, meta_cols: t.Sequence[str]) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    validation = submission_df[submission_df['id'].str.contains('validation')] 
    evaluation = submission_df[submission_df['id'].str.contains('evaluation')]
    validation.columns = ["id"] + [f'd_{num}' for num in range(1914,1942)]
    evaluation.columns = ["id"] + [f'd_{num}' for num in range(1914,1942)]
    evaluation['id'] = evaluation['id'].str.replace('evaluation','validation')
    validation = pd.merge(validation, meta_df, how="left", on="id")
    evaluation = pd.merge(evaluation, meta_df, how="left", on="id")
    melted_validation = pd.melt(
        validation,
        id_vars=meta_cols, 
        var_name='day',
        value_name='count'
    )
    melted_evaluation = pd.melt(
        evaluation,
        id_vars=meta_cols, 
        var_name='day',
        value_name='count'
    )
    melted_validation["use"] = "test1"
    melted_evaluation["use"] = "test2"

    return melted_validation, melted_evaluation


def merge_data(dataset: pd.DataFrame, calendar: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    merged_dataset = pd.merge(
        dataset, 
        calendar, 
        how="left", 
        left_on='day', 
        right_on='d'
    ).drop("d", axis=1)
    merged_dataset = pd.merge(
        merged_dataset, 
        price, 
        how="left",
        on=["store_id", "item_id", "wm_yr_wk"]
    ).fillna("0")
    return merged_dataset
    

def add_trend(dataset: pd.DataFrame) -> pd.DataFrame:
    lags = [7,30]
    dfs = []
    print(dataset)
    for id in dataset["id"].unique():
        _df = dataset[dataset['id'] == id]
        for l in lags:
            _df[f'lag_{l}'] = _df["count"].shift(l)
            dfs.append(_df)
    dataset = pd.concat(dfs, axis=0)
    print(dataset["lag_7"])
    print(dataset["lag_30"])



def main(train_path: str, calendar_path: str, price_path: str, sample_submission_path: str, save_dir: str) -> None:
    train = pd.read_csv(train_path)
    calendar = pd.read_csv(calendar_path)
    price = pd.read_csv(price_path)
    sample_submission = pd.read_csv(sample_submission_path) 

    meta = train.loc[:,TRAIN_META]
    train = train.drop(TRAIN_META, axis=1)
    train = pd.concat([meta, extract_data(train, 2)], axis=1)
    train = train_preprocess(train, TRAIN_META)

    test1, test2 = test_preprocess(sample_submission, meta, TRAIN_META)
    dataset = pd.concat([train, test1, test2], axis=0).reset_index(drop=True)
    dataset = merge_data(dataset, calendar, price)
    dataset = add_trend(dataset)



    






    


if __name__ == "__main__":
    main(TRAIN_PATH, CALENDAR_PATH, PRICE_PATH, SAMPLE_SUBMISSION_PATH, SAVE_DIR)