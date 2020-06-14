import logging
import typing as t
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import optuna
import category_encoders as ce
from optuna.integration import lightgbm_tuner
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

DATA_DIR = "src/sample_data/Kaggle/predict_number_of_views"
TRAIN_DATA = f"{DATA_DIR}/train_data.csv"
TEST_DATA = f"{DATA_DIR}/test_data.csv"


def main():
    raw_train = pd.read_csv(TRAIN_DATA)
    raw_test = pd.read_csv(TEST_DATA)

    print(raw_train, raw_train.columns, raw_train.info())



if __name__ == "__main__":
    main()






