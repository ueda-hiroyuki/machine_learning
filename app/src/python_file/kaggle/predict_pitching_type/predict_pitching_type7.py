import gc
import logging
import collections
import typing as t
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import featuretools as ft
import category_encoders as ce # カテゴリ変数encording用ライブラリ
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.impute import SimpleImputer 
from sklearn.decomposition import PCA
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score


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
EXTERNAL_DATA_DIR = f"{DATA_DIR}/external_data"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
EXTERNAL_1_PATH = f"{EXTERNAL_DATA_DIR}/external_data1.csv"
EXTERNAL_2_PATH = f"{EXTERNAL_DATA_DIR}/external_data2.csv"
EXTERNAL_3_PATH = f"{EXTERNAL_DATA_DIR}/external_data3.csv"
EXTERNAL_4_PATH = f"{EXTERNAL_DATA_DIR}/external_data4.csv"
EXTERNAL_5_PATH = f"{EXTERNAL_DATA_DIR}/external_data5.csv"
EXTERNAL_6_PATH = f"{EXTERNAL_DATA_DIR}/external_data6.csv"
EXTERNAL_7_PATH = f"{EXTERNAL_DATA_DIR}/external_data7.csv"

PITCH_REMOVAL_COLUMNS = ["試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, na_values='不明')
    train_player = pd.read_csv(TRAIN_PLAYER_PATH, na_values='不明')
    test_pitching = pd.read_csv(TEST_PITCH_PATH, na_values='不明')
    test_player = pd.read_csv(TEST_PLAYER_PATH, na_values='不明')
    se_player = pd.read_csv(EXTERNAL_1_PATH, engine="python", na_values='-')
    pa_player = pd.read_csv(EXTERNAL_2_PATH, engine="python", na_values='-')
    pitcher_ability = pd.read_csv(EXTERNAL_3_PATH, engine="python", na_values='-')
    player_ability = pd.read_csv(EXTERNAL_4_PATH, engine="python", na_values='-')
    pitcher_result = pd.read_csv(EXTERNAL_7_PATH, engine="python", na_values='-')
    pitcher_result = pitcher_result

    pitcher_result = pitcher_result.replace({"DeNA": "ＤｅＮＡ"}) 
    train_pitching["usage"] = "train"
    test_pitching["usage"] = "test"
    test_pitching["球種"] = 0

    pitching_data = pd.concat([train_pitching, test_pitching], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)
    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1)
    pitchers_data = player_data[player_data["位置"] == "投手"]

    # print(pitching_data, pitching_data.columns)
    # print(pitchers_data, pitchers_data.columns)
    # print(pitcher_result, pitcher_result.columns)
    print(pitchers_data[pitchers_data["選手名"] == "内海　哲也"])
    print(pitcher_result[pitcher_result["選手名"] == "内海　哲也"])
    # print(pitching_data[pitching_data['投手ID'] == 12107])

    merged_pitchers_data = pd.merge(
        pitchers_data, 
        pitcher_result, 
        how="left", 
        left_on=['チーム名','選手名', '年度'], 
        right_on=['チーム','選手名', '年度'],
    ).drop(['チーム'], axis=1)

    # merged_data = pd.merge(
    #     pitching_data, 
    #     merged_pitchers_data, 
    #     how="left", 
    #     left_on=['年度','投手ID'], 
    #     right_on=['年度','選手ID'],
    # ).drop(['選手ID', '投球位置区域'], axis=1)

    # aaa = merged_data.isna().sum()
    # for i in aaa.items():
    #     print(i)

    


if __name__ == "__main__":
    main()