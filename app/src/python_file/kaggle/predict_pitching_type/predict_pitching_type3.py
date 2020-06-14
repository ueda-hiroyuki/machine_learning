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
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

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

PITCH_REMOVAL_COLUMNS = ["データ内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
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

def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=10000,
    )
    return model_selection

def objective(X, y, trial):
    """最適化する目的関数"""
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        #n_jobs = 4,
        n_estimators=10000,
    )
    n_components = trial.suggest_int('n_components', 1, len(list(X.columns))),
    pca = PCA(n_components=n_components[0]).fit(X)
    x_pca = pca.transform(X)
    tr_x, val_x, tr_y, val_y = train_test_split(x_pca, y, random_state=1)
    gbm.fit(
        tr_x, 
        tr_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=20,
        verbose=50
    )
    y_pred = gbm.predict(val_x)
    accuracy = accuracy_score(val_y, y_pred)
    return accuracy

def get_selected_columns(train_x, train_y):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    est = lgb.LGBMClassifier()
    selector = RFECV(estimator=est, step=0.05, n_jobs=1, min_features_to_select=round(len(train_x)*0.5), cv=skf, verbose=10)
    selector.fit(train_x, train_y)
    selected_columns = train_x.columns[selector.support_]
    return selected_columns


def get_important_features(train_x: t.Any, train_y: t.Any, best_feature_count: int):
    pca = PCA(n_components=best_feature_count).fit(train_x)
    x_pca = pca.transform(train_x)
    return x_pca, train_y



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

    use = merged_data.loc[:, "use"]
    labal = merged_data.loc[:, "球種"]
    merged_data = merged_data.drop(["use", "位置", "球種"], axis=1)

    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute') 
    encorded_data = ce_oe.fit_transform(merged_data) 
    encorded_data = pd.concat([encorded_data, use, labal], axis=1)

    encorded_data = encorded_data[encorded_data["試合ID"] == 2017033101] # 

    lags = [1,2,3]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        encorded_data[lag_col] = 0


    print(encorded_data.info())


 
    # train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    # test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    # train_x = train.drop("球種", axis=1)
    # train_y = train.loc[:,"球種"]
    # test_x = test.drop("球種", axis=1)

    # print(train_x.columns, train_x)



if __name__ == "__main__":
    main()
