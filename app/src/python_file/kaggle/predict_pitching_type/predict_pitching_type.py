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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager


"""
train_pitch(51 columns)
test_pitch(49 columns)
2つの差は「球種」と「投球位置区域」である
⇒今回は「球種」を分類する(0~7の8種類)。

train_player(25 columns)
test_player(25 columns)

・pitchとplayerで共通しているcolumn：「年度」,
・submission.csvに記載する情報は「test_pitchのデータ連番」「各球種(8種)の投球確率」

"""

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/predict_pitching_type"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "年度", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数", "試合ID"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "位置", "出身地", "出身国", "年度", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸"]
LABEL_ENCORDER_COLUMNS = ["球場名", "試合種別詳細", "表裏", "投手投球左右", "投手役割", "打者打席左右", "打者守備位置", "プレイ前走者状況", "チーム名", "選手名", "投", "打", "血液型"]

def remove_columns(df):
    df = df.drop(REMOVAL_COLUMNS, axis=1).fillna(0)
    return df


def label_encorder(col: pd.Series) -> pd.DataFrame:
    if col.dtypes == "object":
        encorded_col = LabelEncoder().fit_transform(col)
        print(encorded_col)
        return encorded_col
    else:
        print(col)
        return col


def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test" 
    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged_data = pd.merge(
        pitch_data, 
        pitchers_data, 
        how="left", 
        left_on='投手ID', 
        right_on='選手ID'
    ).drop(['選手ID', '投球位置区域'], axis=1)
    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use"], axis=1)
    encorded_data = cf.label_encorder(merged_data)
    encorded_data = cf.standardize(encorded_data) # 標準化
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1).reset_index(drop=True)

    print(train_x, train_y, test_x)
  


if __name__ == "__main__":
    main()