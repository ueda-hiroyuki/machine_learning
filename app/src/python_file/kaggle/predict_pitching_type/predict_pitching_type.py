import gc
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager


"""
train_pitch(51 columns)
test_pitch(49 columns)
2つの差は「球種」と「投球位置区域」である
⇒今回は「球種」を予測する。

train_player(25 columns)
test_player(25 columns)

・pitchとplayerで共通しているcolumn：「年度」,
・submission.csvに記載する情報は「test_pitchのデータ連番」「各球種(8種)の投球確率」

"""
# mpl.font_manager._rebuild()
# sns.set(font=['IPAMincho'])
logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/predict_pitching_type"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
REMOVAL_COLUMNS = ["試合内連番", "イニング", "成績対象打者ID", "成績対象投手ID", "チームID", "打者試合内打席数", "試合ID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "投球位置区域"]


def remove_columns(df):
    df = df.drop(REMOVAL_COLUMNS, axis=1).fillna(0)
    return df


def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    pitch_columns = list(train_pitch.columns)
    player_columns = list(train_player.columns)

    # print(pitch_columns)
    # print("###########################")
    # print(player_columns)

    pitchers = train_player[train_player["位置"] == "投手"]
    
    train_data = pd.merge(
        train_pitch, 
        train_player, 
        how="left", 
        left_on='投手ID', 
        right_on='選手ID'
    ).drop(['選手ID'], axis=1)
    train_data = remove_columns(train_data)
    cf.check_corr(train_data, "predict_pitching_type")
    print("####################")
    print(train_data.columns)
    print(train_data)
    print("####################")



if __name__ == "__main__":
    main()