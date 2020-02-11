import matplotlib.pyplot as plt
import pandas as pd
import typing as t
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/items.csv")
    item_categories = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/item_categories.csv")
    df = df.groupby([df.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    df = df[['date','item_id','shop_id','item_cnt_day']]
    df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index() 
    df = df.drop(["2015-11", "2015-12"], axis=1)
    df = pd.merge(items, df, how="right") 
    df = pd.merge(item_categories, df, how="right")
    print(df.columns)
    return df

def remove_outlier(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['item_cnt_day'] < 1000) & (df['item_price'] < 100000)]
    return df

def train_dataset_maker(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train_x_df = df.drop("2015-10", axis=1)
    train_y_df = df.loc[:,"2015-10"]
    return train_x_df, train_y_df

def test_dataset_maker(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/test.csv")
    df = pd.merge(test, df, on=['item_id','shop_id'], how="left").fillna(0)
    id_df = df.loc[:,"ID"]
    df= df.drop(["ID","shop_id","item_id","item_category_name","item_category_id","item_name"], axis=1)
    df = df.drop("2013-01", axis=1)
    return df, id_df

def predict(model: t.Any, test_df: pd.DataFrame) -> t.Any:
    y_pred = model.predict(test_df)
    return y_pred
    
def get_model(train_data: t.Any, val_data: t.Any) -> t.Any:
    lgb_params = {
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 30,
        'min_data_in_leaf': 20,
        'num_iterations' : 100,
        'learning_rate' : 0.5,
        'feature_fraction' : 0.7,
    }
    model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=100,
        valid_sets=val_data,
        early_stopping_rounds=5
    )
    return model

def save_to_csv(df):
    df.to_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/price_pred.csv", index=False)

def main():
    ## parse_dates: 文字列で表示されていた日時をdatetime型として読み込む
    train = pd.read_csv("../sample_data/Kaggle/kaggle_dataset/predict_future_price/sales_train.csv", parse_dates=["date"], infer_datetime_format=True)
    train = remove_outlier(train)
    train = train.drop(["date_block_num","item_price"], axis=1)
    train = preprocess(train)
    print(len(train))
    train_df = train.drop(["shop_id","item_id","item_category_name","item_category_id","item_name"], axis=1)
    train_x_df, train_y_df = train_dataset_maker(train_df)
    test_x_df, id_df = test_dataset_maker(train)

    y_preds = []
    for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(train_x_df, train_y_df):
        tr_x = train_x_df.iloc[train_idx]
        tr_y = train_y_df.iloc[train_idx]
        val_x = train_x_df.iloc[val_idx]
        val_y = train_y_df.iloc[val_idx]

        train_data = lgb.Dataset(tr_x, tr_y)
        val_data = lgb.Dataset(val_x, val_y, reference=train_data)
        model = get_model(train_data, val_data)
        
        y_pred = predict(model, test_x_df)
        y_preds.append(y_pred)
    
    preds_df = pd.DataFrame(y_preds).transpose()
    pred_df = preds_df.mean(axis='columns')
    submission_df = pd.concat([id_df, pred_df], axis=1).rename(columns={0:"item_cnt_month"})
    print(submission_df)
    save_to_csv(submission_df)

if __name__ == "__main__":
    # 2015年11月の売り上げを予測する。
    main()