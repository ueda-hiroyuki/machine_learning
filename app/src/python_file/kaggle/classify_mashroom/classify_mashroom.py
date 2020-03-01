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


TRAIN_PATH = 'src/sample_data/Kaggle/classify_mashroom/train.tsv'
TEST_PATH = 'src/sample_data/Kaggle/classify_mashroom/test.tsv'

CAP_SHAPE = ['x', 'f', 'k', 'f', 'b', 's', 'c']
CAP_SURFACE = ['y', 's', 'f', 'g']
CAP_COLOR = ['n', 'g', 'e', 'w', 'y', 'b', 'p', 'c', 'r', 'u']
BRUISES = ['t', 'f']
ODOR = ['n', 'f', 's', 'y', 'l', 'a', 'p', 'c', 'm']
GILL_ATTACHMENT = ['f', 'a']
GILL_SPACING = ['c', 'w']
GILL_SIZE = ['b', 'n']
GILL_COLOR = ['b', 'p', 'w', 'n', 'g', 'h', 'u', 'k', 'e', 'y', 'o', 'r']
STALK_SHAPE = ['t', 'e']
STALK_ROOT = ['b', 'e', 'c', 'r', '?']
STALK_SURFACE_ABOVE_RING = ['s', 'k', 'f', 'y']
STALK_SURFACE_BELOW_RING = ['s', 'k', 'f', 'y']
STALK_COLOR_ABOVE_RING = ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
STALK_COLOR_BELOW_RING = ['w', 'p', 'g', 'n', 'b', 'o', 'e', 'c', 'y']
VEIL_TYPE = ['p']
VEIL_COLOR = ['w', 'o', 'n', 'y']
RING_NUMBER = ['o', 't', 'n']
RING_TYPE = ['p', 'e', 'l', 'f', 'n']
SPORE_PRINT_COLOR = ['w', 'n', 'k', 'h', 'r', 'u', 'y', 'o', 'b']
POPULATION = ['v', 'y', 's', 'n', 'a', 'c']
HABITAT = ['d', 'g', 'p', 'l', 'u', 'm', 'w']
LABEL = ['e', 'p']

COLUMNS = {
    'cap-shape': CAP_SHAPE,
    'cap-surface': CAP_SURFACE,
    'cap-color': CAP_COLOR,
    'bruises': BRUISES,
    'odor': ODOR,
    'gill-attachment': GILL_ATTACHMENT,
    'gill-spacing': GILL_SPACING,
    'gill-size': GILL_SIZE,
    'gill-color': GILL_COLOR,
    'stalk-shape': STALK_SHAPE,
    'stalk-root': STALK_ROOT,
    'stalk-surface-above-ring': STALK_SURFACE_ABOVE_RING,
    'stalk-surface-below-ring': STALK_SURFACE_BELOW_RING,
    'stalk-color-above-ring': STALK_COLOR_ABOVE_RING,
    'stalk-color-below-ring': STALK_COLOR_BELOW_RING,
    'veil-type': VEIL_TYPE,
    'veil-color': VEIL_COLOR,
    'ring-number': RING_NUMBER,
    'ring-type': RING_TYPE,
    'spore-print-color': SPORE_PRINT_COLOR,
    'population': POPULATION,
    'habitat': HABITAT
}


def preprocessing_train(train: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.Series]:
    train = convert_to_value(train)
    train_y = train.loc[:, 'Y'].replace({'e': 0, 'p': 1})
    train_x = train.drop(['id','Y'], axis=1)
    return train_x ,train_y


def preprocessing_test(test: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.Series]:
    test = convert_to_value(test)
    ids = test.loc[:, 'id']
    test = test.drop(['id'], axis=1)
    print(test)
    return test, ids


def convert_to_value(df: pd.DataFrame) -> pd.DataFrame:
    for key, value in COLUMNS.items():
        for idx, row in enumerate(value):
            df[key] = df[key].mask(df[key] == row, idx)
    return df


def main(train_path: str, test_path: str) -> None:
    train = pd.read_table(train_path)
    test = pd.read_table(test_path)

    train_x, train_y = preprocessing_train(train)
    test, ids = preprocessing_test(test)

    # train_x.apply(lambda x: print(x.value_counts()))
    print(train_x, train_y)
    print(test, ids)
    # print(train_x.dtypes)

if __name__ == "__main__":
    main(TRAIN_PATH, TEST_PATH)