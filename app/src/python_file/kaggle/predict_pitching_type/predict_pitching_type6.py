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
from sklearn.feature_selection import RFE, RFECV
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score, cohen_kappa_score


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

PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8


def main():
    """
    << 処理の流れ >>
    データ読み込み ⇒ 投球データと選手データの結合(train,testも結合) ⇒ nanの置換 ⇒ カテゴリ変数の変換 ⇒
    RFEによる特徴量選択(個数の最適化) ⇒ ハイパーパラメータの最適化 ⇒ 交差検証
    """

    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test"
    test_pitch["球種"] = 0
    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged_data = pd.merge(
        pitch_data, 
        player_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)

    label = merged_data.loc[:, "球種"]
    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use", "位置", "年度", "球種"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_ohe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_ohe.fit_transform(merged_data) 

    for column_name, item in encorded_data.iteritems():
        if item.dtype == "int64":
            encorded_data[column_name] = item.astype('int32')
        else:
            encorded_data[column_name] = item.astype('float32')
        
    encorded_data = pd.concat([encorded_data, use, label], axis=1)
    
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)
    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)


    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        #n_jobs = 4,
        n_estimators=200,
    )
    # RFE で取り出す特徴量の数を最適化する
    rfe = RFE(estimator=gbm, n_features_to_select=100, step=0.05, verbose=10)
    rfe.fit(train_x, train_y)
    selected_columns = list(train_x.columns[rfe.support_])

    selected_train_x = train_x.loc[:, selected_columns]
    selected_test_x = test_x.loc[:, selected_columns]

    tr_x, val_x, tr_y, val_y = train_test_split(selected_train_x, train_y, test_size=0.2, stratify=train_y)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    tr_dataset = lgb.Dataset(tr_x, tr_y)
    val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 8,
        'boosting_type': 'gbdt', 
        'metric': 'multi_logloss'
    }

    model = lgb.train(
        lgb_params,
        tr_dataset,
        valid_sets=val_dataset,
        num_boost_round=1000,
        early_stopping_rounds=10
    )
    y_pred = model.predict(selected_test_x)
    print(y_pred, y_pred.shape)
    submission = pd.DataFrame(y_pred)
    print(submission)
    submission.to_csv(f"{DATA_DIR}/my_submission20.csv", header=False)


if __name__ == "__main__":
    main()
