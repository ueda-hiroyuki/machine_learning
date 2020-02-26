import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TRAIN_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/train.csv'
TEST_PATH = 'src/sample_data/Kaggle/predict_target_of_bank/test.csv'

MARRIED = ['married', 'single', 'divorced']
HOUSING = ['yes', 'no']
COLUMNS = {
    'marital': MARRIED,
    'housing': HOUSING,
}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train_preprocess(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    train_x = df.drop('y', axis=1)
    train_y = df.loc[:,'y']
    train_x = replace_to_value(train_x) 
    return train_x, train_y


def test_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    ...


def replace_to_value(df: pd.DataFrame) -> None:
    for key, value in COLUMNS.items():
        for idx, row in enumerate(value):
            df[key] = df[key].mask(df[key] == row, idx)
    return df


def check_fig(df: pd.DataFrame) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f'src/sample_data/Kaggle/predict_target_of_bank/{name}.png')
        

def main(train_path: str, test_path: str) -> None:
    train = read_csv(train_path)
    test = read_csv(test_path)

    train_x, train_y = train_preprocess(train)
    # test = train_preprocess(test)
    train_x.apply(lambda x: print(x.value_counts(dropna=False)))
    print(train_x.dtypes)
    print(train_x)

if __name__ == "__main__":
    main(TRAIN_PATH, TEST_PATH)