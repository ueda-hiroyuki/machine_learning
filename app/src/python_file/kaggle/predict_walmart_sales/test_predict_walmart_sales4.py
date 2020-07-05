import os
import pandas as pd
import numpy as np
import lightgbm
import category_encoders as ce
import joblib

DATA_DIR = "src/sample_data/Kaggle/predict_walmart_sales"
TRAIN_EVALUATION_PATH = f'{DATA_DIR}/sales_train_evaluation.csv'
TRAIN_VALIDATION_PATH = f'{DATA_DIR}/sales_train_validation.csv'
CALENDAR_PATH = f'{DATA_DIR}/calendar.csv'
PRICE_PATH = f'{DATA_DIR}/sell_prices.csv'
SAMPLE_SUBMISSION_PATH = f'{DATA_DIR}/sample_submission.csv'

FIXED_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

"""
d_1914~d_1941までの28日間の各アイテムの売り上げを予測する
"""

LAGS = [7, 28]
WINDOW_SIZE = [7, 28]
VALID_NUM = 1914
EVAL_NUM = 1942


def add_feature(df, lags, window_size):
    lag_cols = [f'lag_{lag}' for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[["id", "sales"]].groupby("id")["sales"].shift(lag) # idでグループ分けし、salesを7，28日ずつずらして過去のデータを特徴量として入れる
    for win in window_size:
        for lag, lag_col in zip(lags, lag_cols):
            # ラグ変数追加後、各ラグ変数の(7，28日前)平均値を列として追加(7、28日分の平均(win))
            df[f"rmean_{lag}_{win}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(win).mean())
    return df


def get_model(train_set, valid_set):
    params = {
        "num_threads": 2,
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.1,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : 5000,
    }

    model = lgb.train(
        params, 
        train_set, 
        valid_sets=valid_set, 
        verbose_eval=50,
        early_stopping_rounds=10,
    )
    joblib.dump(model, f"{DATA_DIR}/lgb_model.pkl")
    return model


def main():
    sales = pd.read_csv(TRAIN_EVALUATION_PATH)
    sales.name = 'sales' # dataframeの名前を指定している
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar.name = 'calendar'
    prices = pd.read_csv(PRICE_PATH)
    prices.name = 'prices'
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    sample_submission.name = 'submission'
    ids = sample_submission.loc[:,"id"]
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk"]

    # preprocessing calender df
    calendar = calendar.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1)
    ce_oe = ce.OrdinalEncoder(cols=["event_name_1", "event_name_2"], handle_unknown='impute')
    calendar = ce_oe.fit_transform(calendar) 
    calendar = calendar.apply(lambda x : x.astype(np.int8) if x.dtype != "object" else x)

    # preprocessing sales df
    for day in range(EVAL_NUM, EVAL_NUM+28+1): # test用の時は予測する部分のカラム(d_1942~d_1971)を追加し、nanで埋めておく。 
        sales[f"d_{day}"] = np.nan
    sales = pd.melt(
        sales,
        id_vars=FIXED_COLS, # 各日付の部分(カテゴリ変数以外の部分)を縦に並べていく
        var_name = "d",
        value_name = "sales"
    ) 
    sales = add_feature(sales, LAGS, WINDOW_SIZE) # ラグ変数とラグ変数の平均値を特徴量として追加する
    sales['d'] = df["d"].str.split(pat='_')[1]
    # 3つのDataframeを統合する
    merged = pd.merge(
        sales,
        calendar,
        how="left", 
        on="d"
    )
    merged = pd.merge(
       merged,
       prices,
       how="left", 
       on=["store_id", "item_id", "wm_yr_wk"]
    ).dropna()

    # カテゴリ変数をencording
    categorical_cols = [col for col in merged.columns if merged[col].dtype == "object"]
    ce_oe = ce.OrdinalEncoder(cols=categorical_cols, handle_unknown='impute')
    encoded = ce_oe.fit_transform(merged) 

    # 学習用(0~1913)、検証用(1914~1941)、評価用(1942~1970)にデータを分割
    train = encoded[encoded["d"] < 1914]
    valid = encoded[(encoded["d"] >= 1914) & (encoded["d"] < 1942)]
    test = encoded[encoded["d"] >= 1942]

    train_x = train.drop(useless_cols, axis=1)
    train_y = train.loc[:, "sales"]
    valid_x = valid.drop(useless_cols, axis=1)
    valid_y = valid.loc[:, "sales"]
    test_x = test.drop(useless_cols, axis=1)

    if not os.path.isfile(f"{DATA_DIR}/lgb_model.pkl"):
        # データセットの作成
        train_set = lgb.Dataset(train_x, train_y)
        valid_set = lgb.Dataset(valid_x, valid_y, reference=train_set)

        # 学習
        model = get_model(train_set, valid_set)
    else:
        model = joblib.load(f"{DATA_DIR}/lgb_model.pkl")
    # 推論
    valid_pred = model.predict(valid_x)
    test_pred = model.predict(test_x)
    



    
    
     


if __name__ == "__main__":
    main()