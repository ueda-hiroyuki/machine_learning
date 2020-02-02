import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_kesson_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    kesson_table_ren_columns = kesson_table_ren_columns.sort_values('欠損数')
    return kesson_table_ren_columns

def get_histgram(x,y):
    plt.scatter(x,y)
    plt.show()

def preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_id = test["Id"]

    train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["LowQualFinSF"]
    train = train.drop(train[(train['TotalSF']>4000) & (train['SalePrice']<200000)].index)
    train = train.drop(train[(train['TotalSF']>3000) & (train['SalePrice']<210000)].index)
    train = train.drop(train[(train['YearBuilt']<1990) & (train['SalePrice']>200000)].index)
    train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)
    train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)
    train = train.drop(train[(train['OverallQual']>9) & (train['SalePrice']<200000)].index)

    kesson = get_kesson_table(train)
    non_kesson_train = train.dropna(how="any", axis=1)
    # print(pd.DataFrame(non_kesson_train.dtypes))
    non_kesson_train = non_kesson_train.select_dtypes(include=["int64"])
    
    columns = non_kesson_train.columns.tolist()
    columns_series = pd.Series(columns)
    
    corrcoef_df = pd.DataFrame(np.corrcoef(non_kesson_train.transpose()), columns=columns)
    corrcoef = corrcoef_df["SalePrice"]
    
    corrcoef_df = pd.concat([columns_series, corrcoef], axis=1)
    corrcoef_df = corrcoef_df.query("SalePrice > 0.5")
    corrcoef_list = corrcoef_df[0].tolist()
    
    non_kesson_train = non_kesson_train.loc[:,corrcoef_list]
    train_x = non_kesson_train.drop("SalePrice", axis= 1)
    train_y = non_kesson_train["SalePrice"]

    non_kesson_test = test.dropna(how="any", axis=1).loc[:, corrcoef_list].dropna(axis=1)
    test_columns = non_kesson_test.columns.tolist()
    train_x = train_x.loc[:, test_columns]

    return train_x, train_y, non_kesson_test, test_id

def Kfold(fold):
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=0)
    return skf

def train_preprocess(x, y):
    tr_x, val_x, tr_y, val_y = train_test_split(x, y, test_size=0.2, random_state=1)

    train = lgb.Dataset(tr_x, tr_y)
    val = lgb.Dataset(val_x, val_y, reference=train)

    return train, val

def train(train_data, val_data):
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

def predict(model, test_data):
    y_pred = model.predict(test_data, num_iterations=model.best_iteration)
    return y_pred


def save_to_csv(df):
    df.to_csv("../Kaggle/kaggle_dataset/house price/price_pred.csv")
    
def main():
    train_x, train_y, non_kesson_test, test_id = preprocess(
        "../Kaggle/kaggle_dataset/house price/train.csv", 
        "../Kaggle/kaggle_dataset/house price/test.csv"
    )

    skf = Kfold(5)
    y_preds = []
    for train_idx, val_idx in skf.split(train_x, train_y):
        tr_x = train_x.iloc[train_idx]
        tr_y = train_y.iloc[train_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        train_data = lgb.Dataset(tr_x, tr_y)
        val_data = lgb.Dataset(val_x, val_y, reference=train)

        model = train(train_data, val_data)
        
        y_pred = predict(model, non_kesson_test)
        y_preds.append(y_pred)

    pred_df = pd.DataFrame(y_preds).transpose()
    avg_pred = pred_df.mean(axis='columns')

    pred_df = pd.concat([test_id, avg_pred], axis=1).rename(columns = {0 : 'SalePrice'}).set_index("Id")
    print(pred_df)

    save_to_csv(pred_df)
    

if __name__ == "__main__":
    main()
        