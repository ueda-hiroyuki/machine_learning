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
EXTERNAL_1_PATH = f"{EXTERNAL_DATA_DIR}/2017登録投手の昨年球種配分.csv"
EXTERNAL_2_PATH = f"{EXTERNAL_DATA_DIR}/2018登録投手の昨年球種配分.csv"
EXTERNAL_3_PATH = f"{EXTERNAL_DATA_DIR}/2019登録投手の昨年球種配分.csv"
EXTERNAL_4_PATH = f"{EXTERNAL_DATA_DIR}/パワプロ2016-2018の投手能力.csv"
EXTERNAL_5_PATH = f"{EXTERNAL_DATA_DIR}/パワプロ2016-2018の野手能力.csv"
EXTERNAL_6_PATH = f"{EXTERNAL_DATA_DIR}/両リーグの2014～2018までの各投手の年間成績データ.csv"
EXTERNAL_7_PATH = f"{EXTERNAL_DATA_DIR}/両リーグの2014～2018までの各打者の年間成績データ.csv"


PITCH_REMOVAL_COLUMNS = ["データ内連番", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8


def get_best_params(train_x: t.Any, train_y: t.Any, num_class: int) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y)
    best_params = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_class': num_class,
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=10000,
        early_stopping_rounds=20,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history
    )
    return best_params

def get_model(train_x, train_y, valid_x, valid_y, num_class) -> t.Any:
    # 学習用データセット
    train_set = lgb.Dataset(train_x, train_y)
    # 評価用データセット
    valid_set = lgb.Dataset(valid_x, valid_y)
    lgb_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_class': num_class,
        'num_threads': 2,
        'num_leaves' : 30,
        'min_data_in_leaf': 20,
        'num_iterations' : 100,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.8,
    }
    model = lgb.train(
        params=lgb_params,
        train_set=train_set,
        valid_sets=[train_set, valid_set],
        early_stopping_rounds=20,
        num_boost_round=1000
    )
    importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance.head(20))
    return model


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, na_values='不明')
    train_player = pd.read_csv(TRAIN_PLAYER_PATH, na_values='不明')
    test_pitching = pd.read_csv(TEST_PITCH_PATH, na_values='不明')
    test_player = pd.read_csv(TEST_PLAYER_PATH, na_values='不明')

    pitching_type_2016 = pd.read_csv(EXTERNAL_1_PATH)
    pitching_type_2017 = pd.read_csv(EXTERNAL_2_PATH)
    pitching_type_2018 = pd.read_csv(EXTERNAL_3_PATH)

    pitchers_ability = pd.read_csv(EXTERNAL_4_PATH)
    batters_ability = pd.read_csv(EXTERNAL_5_PATH)

    pitchers_results = pd.read_csv(EXTERNAL_6_PATH).drop(["index"], axis=1)
    batters_results = pd.read_csv(EXTERNAL_7_PATH).drop(["index"], axis=1)

    train_pitching["use"] = "train"
    test_pitching["use"] = "test"
    test_pitching["球種"] = 9999
    test_pitching["投球位置区域"] = 9999

    train_pitching = train_pitching.head(1000) # メモリ節約のため
    test_pitching = test_pitching.head(1000) # メモリ節約のため

    # 2016~2018年の投手毎球種割合を結合
    pitching_type_ratio = pd.concat([pitching_type_2016, pitching_type_2017, pitching_type_2018], axis=0)

    pitch_data = pd.concat([train_pitching, test_pitching], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged = pd.merge(
        pitch_data,
        player_data,
        how="left",
        left_on=['年度','投手ID'],
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)

    # データセットと前年度投球球種割合をmergeする
    merged = pd.merge(
        merged,
        pitching_type_ratio,
        how="left",
        left_on=['年度','投手ID', "選手名"],
        right_on=['年度','選手ID', "選手名"]
    ).drop(['選手ID'], axis=1)

    # データセットと前年度投手成績データをmergeする
    pitchers_results = pitchers_results.replace({'チーム': {"DeNA": "ＤｅＮＡ"}})
    pitchers_results["年度"] = pitchers_results["年度"] + 1
    pitchers_results = pitchers_results[pitchers_results["年度"] >= 2017]
    replasce_cols = list(set(pitchers_results.columns) - set(["年度", "チーム", "選手名"]))
    for col in replasce_cols:
        pitchers_results = pitchers_results.rename(columns={col: f'昨年度_投手_{col}'})

    merged = pd.merge(
        merged,
        pitchers_results,
        how="left",
        left_on=['年度','選手名', "チーム名"],
        right_on=['年度','選手名', "チーム"]
    ).drop(['チーム'], axis=1).fillna(0)

    # データセットと前年度打者成績データをmergeする
    batters_results = batters_results.replace({'チーム': {"DeNA": "ＤｅＮＡ"}})
    batters_results["年度"] = batters_results["年度"] + 1
    batters_results = batters_results[batters_results["年度"] >= 2017]
    replasce_cols = list(set(batters_results.columns) - set(["年度", "チーム", "選手名"]))
    for col in replasce_cols:
        batters_results = batters_results.rename(columns={col: f'昨年度_打者_{col}'})

    # batters_resultsには"打者ID"列がないため、mergeするために列を新規作成
    playersID = player_data[['年度', "選手ID", "選手名"]].drop_duplicates().reset_index(drop=True)
    playersID["打者名"] = playersID["選手名"]
    playersID = playersID.drop("選手名", axis=1)
    print(playersID.columns)
    print(merged.columns)
    merged = pd.merge(
        merged,
        playersID,
        how="left",
        left_on=['年度', "打者ID"],
        right_on=['年度', "選手ID"]
    ).drop(['選手ID'], axis=1)
    print(merged)
    print(merged.columns)
    print(merged[merged.isnull().any(axis=1)])







    # batters_results = pd.merge(
    #     batters_results,
    #     playersID,
    #     how="left",
    #     left_on=['選手名', "チーム"],
    #     right_on=['選手名', "チーム名"]
    # ).drop("チーム", axis=1).fillna(0)
    # batters_results["選手ID"] = batters_results["選手ID"].astype(np.int16)
    # print(batters_results["選手ID"])
    # print(batters_results[batters_results.isnull().any(axis=1)])
    # print(batters_results.columns)

    # merged = pd.merge(
    #     merged,
    #     batters_results,
    #     how="left",
    #     left_on=['年度','打者ID', "チーム名"],
    #     right_on=['年度','選手ID', "チーム名"]
    # ).drop(['選手ID'], axis=1)

    # print(merged)
    # print(merged.columns)
    # print(merged[merged.isnull().any(axis=1)])












if __name__ == "__main__":
    main()