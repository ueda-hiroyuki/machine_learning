import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

TRAIN_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/train.csv'
TEST_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/test.csv'
SAVE_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/submission.csv'

MONTH = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
JOB = ['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'housemaid', 'student', 'unemployed', 'unknown']
POUTCOME = ['success', 'failure', 'other', 'unknown']
EDUCATION = ['secondary','tertiary','primary','unknown']
CONTACT = ['cellular', 'telephone', 'unknown']
MARRIED = ['married', 'single', 'divorced']
HOUSING = ['yes', 'no']
LOAN = ['yes', 'no']
DEFAULT = ['yes', 'no']
COLUMNS = {
    'marital': MARRIED,
    'housing': HOUSING,
    'job': JOB,
    'month': MONTH,
    'poutcome': POUTCOME,
    'education': EDUCATION,
    'contact': CONTACT,
    'loan': LOAN,
    'default': DEFAULT,
}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train_preprocess(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train_x = replace_to_value(df) 
    train_x = convert_type(train_x)
    train_x = smoothing(train_x)
    train_y = train_x.loc[:,'y']
    train_x = train_x.drop(['y','id', 'default', 'loan'], axis=1)
    return train_x, train_y


def test_preprocess(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.Series]:
    test = replace_to_value(df)
    test = convert_type(test)
    ids = test['id']
    test = test.drop(['id', 'default', 'loan'], axis=1)
    return test, ids


def replace_to_value(df: pd.DataFrame) ->pd.DataFrame:
    for key, value in COLUMNS.items():
        for idx, row in enumerate(value):
            df[key] = df[key].mask(df[key] == row, idx)
    return df
    

def convert_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(
        {
            'marital': int,
            'housing': int,
            'job': int,
            'month': int,
            'poutcome': int,
            'education': int,
            'contact': int,
            'loan': int,
            'default': int,
        }
    )
    return df

def remove_outlier(column: pd.Series) -> pd.Series:
    z = stats.zscore(column) < 5
    sm_column = column[z]
    return sm_column


def smoothing(df: pd.DataFrame) -> pd.DataFrame:
    print(df)
    df = df.drop(df[(df['balance']>60000)].index)
    df = df.drop(df[(df['duration']>3000)].index)
    df = df.drop(df[(df['previous']>50)].index)
    print(df)
    return df


def check_fig(df: pd.DataFrame) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f'src/sample_data/Kaggle/predict_target_of_bank/{name}.png')


def check_corr(df: pd.DataFrame) -> None:
    corr = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, square=True, annot=True)
    plt.savefig(f'src/sample_data/Kaggle/predict_target_of_bank/corr_heatmap.png')


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df)
    scaler_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaler_df


def minmaxscaler(df: pd.DataFrame) -> pd.DataFrame:
    mm = MinMaxScaler()
    mm.fit(df)
    mm_df = pd.DataFrame(mm.transform(df), columns=df.columns)
    return mm_df


def get_model(tr_dataset: t.Any, val_dataset: t.Any) -> t.Any:
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        'metric' : {'l2'},
        'num_leaves' : 50,
        'min_data_in_leaf': 100,
        'num_iterations' : 100,
        'learning_rate' : 0.2,
        'feature_fraction' : 0.5,
    }
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=val_dataset,
        early_stopping_rounds=5,
    )
    return model


def prediction(model: t.Any, test_df: pd.DataFrame) -> pd.Series:
    y_pred = model.predict(test_df, num_iteration=model.best_iteration)
    return pd.Series(y_pred)
        

def main(train_path: str, test_path: str) -> None:
    train = read_csv(train_path)
    test = read_csv(test_path)

    train_x, train_y = train_preprocess(train)
    test, ids = test_preprocess(test)
    # check_fig(train_x)
    check_corr(train_x)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    y_preds = []
    importances = []
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]
        print(tr_x, tr_y, val_x, val_y)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset)

        importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=["importance"])
        importances.append(importance)

        y_pred = prediction(model, test)
        y_preds.append(y_pred)

    
    preds_df = pd.concat(y_preds, axis=1)
    
    pred_df = preds_df.mean(axis=1)
    submission = pd.concat([ids, pred_df], axis=1)
    print(submission)
    print("####################")
    print(pd.concat(importances, axis=1))
    submission.to_csv(SAVE_PATH, header=False, index=False)



if __name__ == "__main__":
    main(TRAIN_PATH, TEST_PATH)