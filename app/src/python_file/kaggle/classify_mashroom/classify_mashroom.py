import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
from scipy import stats
import seaborn as sns
from toolz import pipe
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_curve


TRAIN_PATH = 'src/sample_data/Kaggle/classify_mashroom/train.tsv'
TEST_PATH = 'src/sample_data/Kaggle/classify_mashroom/test.tsv'
SAVE_DIR = 'src/sample_data/Kaggle/classify_mashroom'

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
    train_x = train.drop(['id','Y', 'gill-attachment','veil-type','veil-color'], axis=1)
    return train_x ,train_y


def preprocessing_test(test: pd.DataFrame) -> t.Tuple[pd.DataFrame, pd.Series]:
    test = convert_to_value(test)
    ids = test.loc[:, 'id']
    test = test.drop(['id', 'gill-attachment','veil-type','veil-color'], axis=1)
    return test, ids


def convert_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(
        {
            'cap-shape': int,
            'cap-surface': int,
            'cap-color': int,
            'bruises': int,
            'odor': int,
            'gill-attachment': int,
            'gill-spacing': int,
            'gill-size': int,
            'gill-color': int,
            'stalk-shape': int,
            'stalk-root': int,
            'stalk-surface-above-ring': int,
            'stalk-surface-below-ring': int,
            'stalk-color-above-ring': int,
            'stalk-color-below-ring': int,
            'veil-type': int,
            'veil-color': int,
            'ring-number': int,
            'ring-type': int,
            'spore-print-color': int,
            'population': int,
            'habitat': int
        }
    )
    return df


def convert_to_value(df: pd.DataFrame) -> pd.DataFrame:
    for key, value in COLUMNS.items():
        for idx, row in enumerate(value):
            df[key] = df[key].mask(df[key] == row, idx)
    df = convert_type(df)
    return df


def check_corr(df: pd.DataFrame) -> None:
    corr = df.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, square=True, annot=True)
    plt.savefig(f'{SAVE_DIR}/corr_heatmap.png')


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
        early_stopping_rounds=10,
    )
    return model


def prediction(model: t.Any, test: pd.DataFrame) -> pd.Series:
    y_pred = model.predict(test, num_iteration=model.best_iteration)
    return pd.DataFrame(y_pred)


def threshold(column: pd.Series, threshold: float) -> pd.Series:
    print(column)
    column = column.apply(lambda x : 'e' if x <= threshold else 'p')
    print(column)
    return column


def main(train_path: str, test_path: str) -> None:
    train = pd.read_table(train_path)
    test = pd.read_table(test_path)

    train_x, train_y = preprocessing_train(train)
    test, ids = preprocessing_test(test)
    #train_x.apply(lambda x: print(x.value_counts()))
    # check_corr(train_x)

    y_preds = []
    importances = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset)

        importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=["importance"])
        importances.append(importance)

        y_pred = prediction(model, test)
        y_preds.append(y_pred)

    preds_df = pd.concat(y_preds, axis=1)
    pred_df = preds_df.mean(axis=1)
    pred_df = threshold(pred_df, 0.5)

    submission = pd.concat([ids, pred_df], axis=1)
    print(submission)
    print("####################")
    print(pd.concat(importances, axis=1))
    submission.to_csv(f'{SAVE_DIR}/submission.csv', header=False, index=False)

        




if __name__ == "__main__":
    main(TRAIN_PATH, TEST_PATH)