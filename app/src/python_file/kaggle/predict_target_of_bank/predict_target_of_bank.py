import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TRAIN_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/train.csv'
TEST_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/test.csv'

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
    train_x = df.drop('y', axis=1)
    train_y = df.loc[:,'y']
    train_x = replace_to_value(train_x) 

    train_x = convert_type(train_x)
    return train_x, train_y


def test_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    test = replace_to_value(df)
    test = convert_type(test)
    return test


def replace_to_value(df: pd.DataFrame) -> None:
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


def check_fig(df: pd.DataFrame) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f'src/sample_data/Kaggle/predict_target_of_bank/{name}.png')


def get_model(tr_dataset: t.Any, val_dataset: t.Any) -> t.Any:
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        'metric' : {'l2'},
        'num_leaves' : 5,
        'min_data_in_leaf': 5,
        'num_iterations' : 50,
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


def prediction()
        

def main(train_path: str, test_path: str) -> None:
    train = read_csv(train_path)
    test = read_csv(test_path)

    train_x, train_y = train_preprocess(train)
    test = test_preprocess(test)
    train_x.apply(lambda x: print(x.value_counts(dropna=False)))
    print(test.dtypes)
    print(test)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    models = []
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]
        print(tr_x, tr_y, val_x, val_y)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset)

        y_pred = prediction(model, test) 



if __name__ == "__main__":
    main(TRAIN_PATH, TEST_PATH)