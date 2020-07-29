import gc
import os
import logging
import joblib
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
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold, cross_val_predict, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor 
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


PITCH_REMOVAL_COLUMNS = ["試合ID", "時刻", "データ内連番", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8

DEPTH_NUMS = [3,5,10,20,30]

FINAL_REMOVAL_COLUMNS = [
    "試合ID",
    "年度",
    "イニング",
    "ホームチームID",
    "アウェイチームID",
    "投",
    "打",
    "打席内投球数",
    "投手役割",
    "投手試合内対戦打者数"
]

def preprocessing(df):
    #df['走者'] = np.where(df["プレイ前走者状況"] == "___", 0, 1) # プレイ前ランナーがいるかいないか。
    df['BMI'] = (df["体重"]/(df["身長"]/100)**2) # 身長体重をBMIに変換
    df = df.drop(["体重", "身長"], axis=1)
    return df


def get_model(tr_x, tr_y):
    model = MLPClassifier(
        activation='relu', 
        batch_size='auto', 
        early_stopping=True, 
        epsilon=1e-08,
        hidden_layer_sizes=(100,100,100,100,100), 
        learning_rate='constant',
        learning_rate_init=0.05, 
        max_iter=100, 
        momentum=0.9,
        n_iter_no_change=10,
        random_state=1, 
        shuffle=True, 
        solver='sgd', 
        verbose=10,
    )
    model.fit(tr_x, tr_y)
    return model

def get_rf_model(tr_x, tr_y, depth):
    model = RandomForestClassifier(
        bootstrap=True,  
        criterion='gini',
        max_depth=depth, 
        max_features='auto', 
        min_impurity_split=1e-07, 
        min_samples_leaf=1,
        n_estimators=100, 
        n_jobs=4, 
        verbose=10, 
    )
    model.fit(tr_x, tr_y)
    return model


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, parse_dates=["日付"])
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitching = pd.read_csv(TEST_PITCH_PATH, parse_dates=["日付"])
    test_player = pd.read_csv(TEST_PLAYER_PATH)

    pitching_type_2016 = pd.read_csv(EXTERNAL_1_PATH)
    pitching_type_2017 = pd.read_csv(EXTERNAL_2_PATH)
    pitching_type_2018 = pd.read_csv(EXTERNAL_3_PATH)

    train_pitching["use"] = "train"
    test_pitching["use"] = "test"
    test_pitching["球種"] = 9999
    test_pitching["投球位置区域"] = 9999

    # train_pitching = train_pitching.head(10000) # メモリ節約のため
    # test_pitching = test_pitching.head(10000) # メモリ節約のため

    # 2016~2018年の投手毎球種割合を結合
    pitching_type_ratio = pd.concat([pitching_type_2016, pitching_type_2017, pitching_type_2018], axis=0).reset_index(drop=True)

    pitch_data = pd.concat([train_pitching, test_pitching], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1).reset_index(drop=True)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1).reset_index(drop=True) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged = pd.merge(
        pitch_data,
        player_data,
        how="left",
        left_on=['年度','投手ID'],
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)
    merged = merged.rename(columns={"選手名": "投手名", "チーム名": "投手チーム名"})

    # データセットと前年度投球球種割合をmergeする
    merged = pd.merge(
        merged,
        pitching_type_ratio,
        how="left",
        left_on=['年度','投手ID', "投手名"],
        right_on=['年度','選手ID', "選手名"]
    ).drop(['選手ID', "選手名"], axis=1)

    use = merged.loc[:, "use"]
    label = merged.loc[:, "球種"]
    merged = merged.drop(["use", "球種", "位置", "年度", "投手名", "日付"], axis=1)

    merged = preprocessing(merged)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged.columns if merged[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged) 
    # encorded_data = cf.standardize(encorded_data)

    encorded_data = pd.concat([encorded_data, use, label], axis=1)
    print(encorded_data)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)
    train_x = train.drop(["球種"], axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop(["球種"], axis=1)

    # train_x_resampled, train_y_resampled = gen_resampled_data(train_x, train_y)
    pseudo_train_x, pseudo_train_y = train_x, train_y
    num_class = 8

    n_splits = 5
    for depth, num in zip(DEPTH_NUMS, range(52, 57)):
        submission = np.zeros((len(test_x),num_class))
        print("################################")
        print(f"start {depth} depth !!")
        print("################################")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for i, (tr_idx, val_idx) in enumerate(skf.split(pseudo_train_x, pseudo_train_y)):
            tr_x = pseudo_train_x.iloc[tr_idx].reset_index(drop=True)
            tr_y = pseudo_train_y.iloc[tr_idx].reset_index(drop=True)
            
            model = get_rf_model(tr_x, tr_y, depth)
            y_preda = model.predict_proba(test_x)
            submission += y_preda

        submission_df = pd.DataFrame(submission)/n_splits
        submission_df.to_csv(f"{DATA_DIR}/my_submission{num}.csv", header=False)
        print("#################################")
        print(submission_df)
        print("#################################")


    


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executer:
        executer.submit(main()) # CPU4つ使っている。