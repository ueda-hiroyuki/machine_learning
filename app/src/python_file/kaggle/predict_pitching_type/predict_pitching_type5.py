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
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, TimeSeriesSplit
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

estimators = [
    ("lgbc1",lgb.LGBMClassifier(random_state=1)),
    # ("lgbc2",lgb.LGBMClassifier(random_state=10)),
    # ("lgbc3",lgb.LGBMClassifier(random_state=100)),
    # ("lgbc4",lgb.LGBMClassifier(random_state=1000)),
    # ("lgbc5",lgb.LGBMClassifier(random_state=10000))   
]

NUM_CLASS = 8


def preprocessing(df):
    #df['走者'] = np.where(df["プレイ前走者状況"] == "___", 0, 1) # プレイ前ランナーがいるかいないか。
    df['BMI'] = (df["体重"]/(df["身長"]/100)**2) # 身長体重をBMIに変換
    df = df.drop(["体重", "身長"], axis=1)
    return df


def gen_adversarital_data(train_x, test_x):
    # Adversarial Validation
    train_z = pd.Series(np.zeros(len(train_x)), name="y")  # 片方は全て 0
    test_z = pd.Series(np.ones(len(test_x)), name="y")  # もう片方は全て 1

    adv_data = pd.concat([train_x, test_x], axis=0).reset_index(drop=True)
    adv_label = pd.concat([train_z, test_z], axis=0).reset_index(drop=True)

    # 要素が最初のデータセット由来なのか、次のデータセット由来なのか分類する
    clf = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    # 分類して確率を計算する
    z_pred_proba = cross_val_predict(
        clf, 
        adv_data,
        adv_label,
        cv=3, 
        method='predict_proba'
    )
    preda_df = pd.DataFrame(z_pred_proba, columns=["train", "test"])
    adv_train = preda_df[:len(train_x)] # train側の予測確率を分離
    adv_test = preda_df[len(train_x):] # test側の予測確率を分離
    print(adv_train)
    pred_train_idx = adv_train[adv_train["test"] > 0.6].index # trainデータの中でテストっぽいデータだと判断された行のindex
    return pred_train_idx



def get_selected_columns(train_x, train_y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    est = lgb.LGBMClassifier()
    selector = RFECV(estimator=est, step=0.05, n_jobs=2, min_features_to_select=round(len(train_x)*0.8), cv=skf, verbose=10)
    selector.fit(train_x, train_y)
    selected_columns = train_x.columns[selector.support_]
    return selected_columns


def get_model(train_x, train_y, valid_x, valid_y, num_class, best_params) -> t.Any:
    # 学習用データセット
    train_set = lgb.Dataset(train_x, train_y)
    # 評価用データセット
    valid_set = lgb.Dataset(valid_x, valid_y)
    # lgb_params = {
    #     'objective': 'multiclass',
    #     'metric': 'multi_logloss',
    #     'boosting_type': 'gbdt',
    #     'num_class': num_class,
    #     'num_threads': 2,
    #     'num_leaves' : 30,
    #     'min_data_in_leaf': 20,
    #     'learning_rate' : 0.1,
    #     'feature_fraction' : 0.8,
    # }
    model = lgb.train(
        params=best_params,
        train_set=train_set,
        valid_sets=[train_set, valid_set],
        early_stopping_rounds=20,
        num_boost_round=1000
    )
    importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance.head(50))
    return model


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, parse_dates=["日付"])
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitching = pd.read_csv(TEST_PITCH_PATH, parse_dates=["日付"])
    test_player = pd.read_csv(TEST_PLAYER_PATH)

    train_pitching["use"] = "train"
    test_pitching["use"] = "test"
    test_pitching["球種"] = 9999
    test_pitching["投球位置区域"] = 9999

    # train_pitching = train_pitching.head(10000) # メモリ節約のため
    # test_pitching = test_pitching.head(10000) # メモリ節約のため

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

    use = merged.loc[:, "use"]
    merged = merged.drop(["use", "位置", "年度"], axis=1)

    merged = preprocessing(merged)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged.columns if merged[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged) 
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)
    train_x = train.drop(["日付", "球種"], axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop(["日付", "球種"], axis=1)

    adv_count = 3
    for i in range(adv_count):
        if i == 0:
            pred_train_idx = gen_adversarital_data(train_x, test_x)
            pred_train_x = train_x.iloc[pred_train_idx].reset_index(drop=True)
            pred_train_y = train_y.iloc[pred_train_idx].reset_index(drop=True)
            #print(pred_train_x, pred_train_y)
        else:
            pred_train_idx = gen_adversarital_data(pred_train_x, test_x)
            pred_train_x = pred_train_x.iloc[pred_train_idx].reset_index(drop=True)
            pred_train_y = pred_train_y.iloc[pred_train_idx].reset_index(drop=True)
            #print(pred_train_x, pred_train_y)
    print("######################################")
    print(pred_train_x, pred_train_y)

    # 先に生成したデータに対応する部分だけ取り出す
    # z_train_pred_proba = z_pred_proba[:2500]

    # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # adv_idx = []
    # for fold, (train_idx, val_idx) in enumerate(skf.split(adv_data, adv_label)):
    #     adv_train_x = adv_data.iloc[train_idx]
    #     adv_valid_x = adv_data.iloc[val_idx]
    #     adv_train_y = adv_label.iloc[train_idx]
    #     adv_valid_y = adv_label.iloc[val_idx]
    #     params = {
    #         'objective': 'binary',
    #         'max_depth': 5,
    #         'boosting': 'gbdt',
    #         'metric': 'auc'
    #     }
    #     train_set = lgb.Dataset(adv_train_x, label=adv_train_y)
    #     model = lgb.train(params, train_set)

    #     y_preda = model.predict(adv_valid_x)
    #     preda_df = pd.Series(y_preda).sort_values(ascending=False)
    #     print(preda_df[preda_df>0.8].index)
    #     adv_idx += list(preda_df[preda_df>0.8].index) 
    # adv_idx = list(set(adv_idx))
    # print(min(adv_idx))
    # print(max(adv_idx))
    # print(len(adv_idx))
    # print(len([i for i in adv_idx if i < len(train_x)]))
    # print(len(train_x))
    # # print([i for i in adv_idx if i < len(train_x)])

    # print(train_x.iloc[adv_idx])



        



if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executer:
        executer.submit(main()) # CPU4つ使っている。