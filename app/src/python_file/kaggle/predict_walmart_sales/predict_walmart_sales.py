import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from kaggle.common import common_funcs as cf

TRAIN_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sales_train_validation.csv'
CALENDAR_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/calendar.csv'
PRICE_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/sell_prices.csv'
SAVE_PATH = 'src/sample_data/Kaggle/kaggle_dataset/predict_walmart_sales/submission.csv'



def main(train_path: str, calendar_path: str, price_path: str, save_path: str) -> None:
    train = pd.read_csv(train_path)
    calendar = pd.read_csv(calendar_path)
    price = pd.read_csv(calendar_path)

    print(train.head(20))

    


if __name__ == "__main__":
    main(TRAIN_PATH, CALENDAR_PATH, PRICE_PATH, SAVE_PATH)