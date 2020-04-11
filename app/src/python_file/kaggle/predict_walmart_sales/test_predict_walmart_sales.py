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
from datetime import datetime, timedelta

TRAIN_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sales_train_validation.csv'
CALENDAR_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/calendar.csv'
PRICE_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sell_prices.csv'
SAMPLE_SUBMISSION_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sample_submission.csv'
SAVE_DIR = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales'

TRAIN_META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
ADD_COLS = ["day", "date", "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"] 
USELESS_COLS = ["id", "date", "count","day", "wm_yr_wk", "weekday", "date", "use"]

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
"event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
"month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
h = 28 
max_lags = 57
tr_last = 1913
FIRST_DAY = 350
fday = datetime(2016,4, 25)

def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv(PRICE_PATH, dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv(CALENDAR_PATH, dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(TRAIN_PATH, nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    dt = pd.melt(
        dt,
        id_vars = catcols,
        value_vars = [col for col in dt.columns if col.startswith("d_")],
        var_name = "d",
        value_name = "sales")
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    return dt


def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


def main() -> None:
    df = create_dt(is_train=True, first_day= FIRST_DAY)
    create_fea(df)
    df.dropna(inplace = True)
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
    X_train = df[train_cols]
    y_train = df["sales"]
    np.random.seed(777)
    fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], categorical_feature=cat_feats, free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],categorical_feature=cat_feats,free_raw_data=False)
    params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.05,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 2000,
        'num_leaves': 150,
        "min_data_in_leaf": 50,
    }
    m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=50) 
    alphas = [1.035, 1.03, 1.025]
    weights = [1/len(alphas)]*len(alphas)
    sub = 0.

    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        te = create_dt(False)
        cols = [f"F{i}" for i in range(1,29)]

        for tdelta in range(0, 28):
            day = fday + timedelta(days=tdelta)
            print(icount, day)
            tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
            create_fea(tst)
            tst = tst.loc[tst.date == day , train_cols]
            te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst)
        te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
        te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace = True)
        te_sub.sort_values("id", inplace = True)
        te_sub.reset_index(drop=True, inplace = True)
        te_sub.to_csv(f"submission_{icount}.csv",index=False)
        if icount == 0 :
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols]*weight
        print(icount, alpha, weight)
    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    sub.to_csv("submission.csv",index=False)



if __name__ == "__main__":
    main()