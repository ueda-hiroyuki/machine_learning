import matplotlib.pyplot as plt
import pandas as pd
import typing as t
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.read_csv("src/sample_data/Kaggle/kaggle_dataset/predict_future_price/items.csv")
    item_categories = pd.read_csv("src/sample_data/Kaggle/kaggle_dataset/predict_future_price/item_categories.csv")
    df = df.groupby([df.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
    df = df[['date','item_id','shop_id','item_cnt_day']]
    df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index() 
    df = df.drop(["2015-11", "2015-12"], axis=1)
    df = pd.merge(items, df, how="right") 
    df = pd.merge(item_categories, df, how="right")
    return df

def remove_outlier(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['item_cnt_day'] < 1000) & (df['item_price'] < 100000)]
    return df

def train_dataset_maker(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train_x_df = df.drop("2015-10", axis=1)
    train_y_df = df.loc[:,"2015-10"]
    return train_x_df, train_y_df

def test_dataset_maker(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv("src/sample_data/Kaggle/kaggle_dataset/predict_future_price/test.csv")
    df = pd.merge(test, df, on=['item_id','shop_id'], how="left").fillna(0)
    id_df = df.loc[:,"ID"]
    df= df.drop(["ID","shop_id","item_id","item_category_name","item_category_id","item_name"], axis=1)
    df = df.drop("2013-01", axis=1)
    return df, id_df

def predict(model: t.Any, test_df: pd.DataFrame) -> t.Any:
    y_pred = model.predict(test_df)
    return y_pred
    
def get_model(train_data: t.Any, valid_data: t.Any) -> t.Any:
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
        valid_sets=valid_data,
        num_boost_round=100,
    )
    return model

def save_to_csv(df):
    df.to_csv("src/sample_data/Kaggle/kaggle_dataset/predict_future_price/price_pred.csv", index=False)

def main():
    ## parse_dates: 文字列で表示されていた日時をdatetime型として読み込む
    df = pd.read_csv("src/sample_data/Kaggle/kaggle_dataset/predict_future_price/sales_train.csv", parse_dates=["date"], infer_datetime_format=True)
    df = remove_outlier(df)
    df = df.drop(["date_block_num","item_price"], axis=1)
    df = preprocess(df)
    df = df.drop(["shop_id","item_id","item_category_name","item_category_id","item_name"], axis=1)

    col = [col for col in df.columns if '2015' not in col]
    train_df = df[col]
    test_df = df.filter(regex='2015.*').iloc[:,4:10]

    print(train_df)
    print(test_df)

    models = []
    for i in range(0,9):
        shift_df = train_df.shift(-i*2, axis=1).iloc[:, 0:7]
        print(f"############{i}############")
        train_x_df = shift_df.iloc[:, 0:6]
        train_y_df = shift_df.iloc[:, [6]]
        print(shift_df,train_x_df,train_y_df)
        train_data = lgb.Dataset(train_x_df)
        valid_data = lgb.Dataset(train_y_df)
        model = get_model(train_data, valid_data)
        models.append(model)

    y_preds = []
    for model in models:
        y_pred = predict(model, test_df)
        y_preds.append(y_pred)
    pred_df = pd.DataFrame(y_preds)
    print(pred_df)



    # tscv = TimeSeriesSplit(n_splits=5)
    # for train_index, val_index in tscv.split(train_x_df.T):
    #     print("---------------------------------")
    #     print("TRAIN:", train_index, "TEST:", val_index)
    #     print("---------------------------------")
    #     train = train_x_df.iloc[:,train_index]
    #     val = train_x_df.iloc[:,val_index]
    #     print("---------------------------------")
    #     print(train, val)
    #     print("---------------------------------")
    #     model = lgb.LGBMRegressor(
    #         n_estimators=200,
    #         learning_rate=0.7,
    #         num_leaves=32,
    #         max_depth=8,
    #         min_child_weight=40
    #     )
    #     model.fit()






    # model = lgb.LGBMRegressor(
    #     boosting_type='gbdt', 
    #     num_leaves=31, 
    #     max_depth=5, 
    #     learning_rate=0.7
    # )
    # model.fit(train_x_df, train_y_df)
    # y_pred = predict(model, test_x_df)
    
    # pred_df = pd.DataFrame(y_pred, columns=['item_cnt_month'])
    # pred_df.to_csv('src/sample_data/Kaggle/kaggle_dataset/predict_future_price/price_pred.csv', index_label='ID')

if __name__ == "__main__":
    # 2015年11月の売り上げを予測する。
    main()