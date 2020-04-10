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
USELESS_COLS = ["id", "date", "count","day", "wm_yr_wk", "weekday", "date", "use"]

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


def test_preprocess(submission_df: pd.DataFrame, meta_df: pd.DataFrame, meta_cols: t.Sequence[str]) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = submission_df[submission_df['id'].str.contains('validation')] 
    evaluation = submission_df[submission_df['id'].str.contains('evaluation')]
    test_id_df = validation.loc[:,"id"]
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

    return melted_validation, melted_evaluation, test_id_df


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
    ).fillna("0").astype({'sell_price': float})
    return merged_dataset
    

def add_trend(dataset: pd.DataFrame) -> pd.DataFrame:
    lags = [7, 30]
    window_sizes = [7, 30]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dataset[lag_col] = dataset[["id","count"]].groupby("id")["count"].shift(lag)

    for window_size in window_sizes :
        for lag,lag_col in zip(lags, lag_cols):
            dataset[f"rmean_{lag}_{window_size}"] = dataset[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(window_size).mean())
    return dataset.dropna()


def get_model(train_dataset: t.Any, valid_dataset: t.Any) -> t.Any:
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        'metric' : {'l2'},
        'num_leaves' : 200,
        'min_data_in_leaf': 1000,
        'num_iterations' : 100,
        'learning_rate' : 0.5,
        'feature_fraction' : 0.6,
    }
    model = lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=valid_dataset,
        early_stopping_rounds=10,
    )
    return model


def predict_label(model: t.Any, test_df: pd.DataFrame) -> pd.Series:
    y_pred = model.predict(test_df, num_iteration=model.best_iteration)
    return pd.Series(y_pred)


def main(train_path: str, calendar_path: str, price_path: str, sample_submission_path: str, save_dir: str) -> None:
    train = pd.read_csv(train_path)
    calendar = pd.read_csv(calendar_path)
    price = pd.read_csv(price_path)
    sample_submission = pd.read_csv(sample_submission_path) 

    meta = train.loc[:,TRAIN_META]
    train = train.drop(TRAIN_META, axis=1)
    train = pd.concat([meta, extract_data(train,730)], axis=1)
    train = train_preprocess(train, TRAIN_META)

    test1, test2, test_id_df = test_preprocess(sample_submission, meta, TRAIN_META)
    dataset = pd.concat([train, test1, test2], axis=0).reset_index(drop=True)
    dataset = merge_data(dataset, calendar, price)
    dataset = add_trend(dataset)
    print(dataset)

    dataset = cf.label_encorder(dataset, [*TRAIN_META, *ADD_COLS])
    
    train = dataset[dataset["use"] == "train"]
    valid = dataset[dataset["use"] == "test1"]
    test = dataset[dataset["use"] == "test2"]
    train_x = train.drop(USELESS_COLS, axis=1)
    train_y = train.loc[:,"count"]
    valid_x = valid.drop(USELESS_COLS, axis=1)
    valid_y = valid.loc[:,"count"]
    test_x = test.drop(USELESS_COLS, axis=1)

    print(train_x)


    train_dataset = lgb.Dataset(train_x, train_y)
    valid_dataset = lgb.Dataset(valid_x, valid_y, reference=train_dataset)

    model = get_model(train_dataset, valid_dataset)
    y_pred = predict_label(model, test_x)

    print(y_pred)    
    sub_cols = [f"F{idx}" for idx in range(1,29)]
    pred_df = pd.DataFrame(y_pred.to_numpy(dtype=float).reshape(int(len(y_pred)/28), 28), columns=sub_cols)
    print(pred_df)
    validation_ids = test_id_df
    evaluation_ids = test_id_df.str.replace('validation','evaluation')

    val_pred = pd.concat([test_id_df, pred_df], axis=1)
    eval_pred = pd.concat([test_id_df.str.replace('validation','evaluation'), pred_df], axis=1)
    sub_df = pd.concat([val_pred, eval_pred], axis=0)
    print(sub_df)
    sub_df.to_csv(f"{SAVE_DIR}/submission1.csv", index=False)


if __name__ == "__main__":
    main(TRAIN_PATH, CALENDAR_PATH, PRICE_PATH, SAMPLE_SUBMISSION_PATH, SAVE_DIR)