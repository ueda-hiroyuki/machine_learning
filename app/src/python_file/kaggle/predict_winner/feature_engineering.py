import itertools
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import (
    LabelEncoder,
    LabelBinarizer,
    OrdinalEncoder,
    MultiLabelBinarizer,
)


weapons = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
]

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"

train_raw_data = pd.read_csv(TRAIN_PATH)
test_raw_data = pd.read_csv(TEST_PATH)

test_raw_data["y"] = 0
train_raw_data["usage"] = 0  # for train
test_raw_data["usage"] = 1  # for test
raw_data = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)


def encoding_method():
    df = train_raw_data.loc[:, weapons].head(3)

    mlb = MultiLabelBinarizer()
    result = mlb.fit_transform(df.values)
    df_trans = pd.DataFrame(result, columns=mlb.classes_)
    print(df)
    print(df_trans)


if __name__ == "__main__":
    encoding_method()