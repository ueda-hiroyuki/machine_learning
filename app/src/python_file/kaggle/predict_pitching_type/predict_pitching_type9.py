import os
import logging
import joblib
import typing as t
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import matplotlib.pyplot as plt
import category_encoders as ce # カテゴリ変数encording用ライブラリ
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.impute import SimpleImputer 
from sklearn.decomposition import PCA
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from python_file.kaggle.common import common_funcs as cf
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

PITCH_REMOVAL_COLUMNS = ["試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8


PIPELINES = {
    'knn': Pipeline([
        ('scl',StandardScaler()),
        ('est',KNeighborsClassifier())
    ]),
    'logistic': Pipeline([
        ('scl',StandardScaler()),
        ('est',LogisticRegression(random_state=1))
    ]),
    'gbm': Pipeline([
        (
            'est',
            lgb.LGBMClassifier(
                objective="multiclass",
                boosting_type= 'gbdt', 
                n_jobs = 4,
            )
        )
    ]),
    'SVC': Pipeline([
        ('scl',StandardScaler()),
        ('est',SVC(random_state=1))
    ]),
}

GRID_SEARCH_PARAMS = {
    'knn':{
        'est__n_neighbors':[2,3,4],
        'est__weights':['uniform','distance'],
        'est__algorithm':['auto'],
        'est__leaf_size':[10,100],
        'est__p':[1,2]
    },
    'logistic': {
        "est__C":[0.1, 0.2, 0.5,  1],
        "est__penalty":['l1', 'l2'],
        'est__class_weight':['balanced'],
        'est__max_iter':[1000, 2000]
    },
    'gbm':{
        'est__num_leaves':[30, 50],
        'est__learning_rate':[0.01, 0.1],
        'est__max_depth':[5, 10],
        'est__n_estimators':[1000, 2000],
    },
    'SVC':{
        "est__C":[0.1, 0.5,  1],
        'est__class_weight':['balanced'],
        'est__max_iter':[1000, 2000]
    },
}

def ckeck_tsne(df: pd.DataFrame) -> None:
    labels = df.loc[:, "球種"]
    df = df.drop("球種", axis=1)
    df_reduced = TSNE(n_components=2, random_state=0).fit_transform(df)
    df_reduced = pd.concat([df_reduced, labels], axis=1)
    plt.figure()
    for t in list(labels.unique()):
        _df = df_reduced[df_reduced["球種"] == t]
        plt.scatter(_df[:, 0], _df[:, 1], label=t)
    plt.legend()
    plt.savefig(f'{DATA_DIR}/tsne_map.png')


def preprocessing(df):
    df['走者'] = np.where(df["プレイ前走者状況"] == "___", 0, 1) # プレイ前ランナーがいるかいないか。
    df['BMI'] = (df["体重"]/(df["身長"]/100)**2) # 身長体重をBMIに変換
    df.drop(["体重", "身長"], axis=1)
    return df


def add_lag(df, lags=[1,2,3]):
    for lag in lags:
        df[f"lag_per_match_{lag}"] = df.groupby(["年度", "試合ID", "投手チームID", "投手ID"])["球種"].shift(lag)
        df[f"lag_per_inning_{lag}"] = df.groupby(["年度", "試合ID", "投手チームID", "イニング", "投手ID"])["球種"].shift(lag)
        df[f"lag_per_bat_{lag}"] = df.groupby(["年度", "試合ID", "投手チームID", "イニング", "イニング内打席数", "投手ID"])["球種"].shift(lag)
    return df

def add_trend(df, lags=[1,2,3]):
    df["mean_per_match"] = df.groupby(["年度", "試合ID", "投手チームID", "投手ID"])["球種"].transform('mean')
    df["mean_per_inning"] = df.groupby(["年度", "試合ID", "投手チームID", "イニング", "投手ID"])["球種"].transform('mean')
    df["mean_per_bat"] = df.groupby(["年度", "試合ID", "投手チームID", "イニング", "イニング内打席数", "投手ID"])["球種"].transform('mean')

    return df


def get_important_features(train_x: t.Any, train_y: t.Any, best_feature_count: int):
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        n_jobs = 4,
    )
    selector = RFE(gbm, n_features_to_select=best_feature_count)
    selector.fit(train_x, train_y) # 学習データを渡す
    selected_train_x = pd.DataFrame(selector.transform(train_x), columns=train_x.columns[selector.support_])
    return selected_train_x, train_y


# optunaによるハイパラの最適化
def get_tuned_model(train_x, train_y, valid_x, valid_y, num_class) -> t.Any:
    # 学習用データセット
    train_set = lgb.Dataset(train_x, train_y)
    # 評価用データセット
    valid_set = lgb.Dataset(valid_x, valid_y)
    # チューニングしたハイパラを格納
    best_params = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'num_class': num_class,
        'num_threads': 2
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        train_set,
        valid_sets=[train_set, valid_set],
        num_boost_round=10000,
        early_stopping_rounds=20,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history
    )
    joblib.dump(gbm, f"{DATA_DIR}/lgb_model.pkl")
    importance = pd.DataFrame(gbm.feature_importance(), index=train_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance)
    return gbm


def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH, parse_dates=["日付"])
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH, parse_dates=["日付"])
    test_player = pd.read_csv(TEST_PLAYER_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test"
    test_pitch["球種"] = 9999
    test_pitch["投球位置区域"] = 9999

    # train_pitch = train_pitch.head(1000) # メモリ節約のため
    # test_pitch = test_pitch.head(1000) # メモリ節約のため

    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged = pd.merge(
        pitch_data, 
        player_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)

    date = merged.loc[:, "日付"]
    usage = merged.loc[:, "use"]
    labal = merged.loc[:, "球種"]
    merged = merged.drop(["日付", "use", "位置", "球種"], axis=1)

    merged = preprocessing(merged)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged.columns if merged[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded = ce_oe.fit_transform(merged) 
    encorded = pd.concat([encorded, date, usage, labal], axis=1)

    lags = [1,2,3] # ラグ特徴量の追加
    encorded = add_trend(encorded)
    encorded = add_lag(encorded, lags)

    encorded = encorded.fillna(encorded.mean()) # 決定木系以外はNanがあると学習できないため、各列の平均値でNanを置換

    train = encorded[(encorded["use"] == "train") & (encorded["日付"] < "2017-9-1")].drop(["use","日付"], axis=1).reset_index(drop=True)
    valid = encorded[(encorded["use"] == "train") & (encorded["日付"] >= "2017-9-1")].drop(["use","日付"], axis=1).reset_index(drop=True)
    test = encorded[(encorded["use"] == "test")].drop(["use","日付"], axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    valid_x = valid.drop("球種", axis=1)
    valid_y = valid.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    if not os.path.isfile(f"{DATA_DIR}/best_estimetors.pkl"):
        best_estimetors = {}
        model_names = [c for c in PIPELINES]
        tscv = TimeSeriesSplit(n_splits=3)
        for (param_name, param), (pipeline_name, pipeline) in zip(GRID_SEARCH_PARAMS.items(), PIPELINES.items()):
            logging.info(f'{param_name} GRID SEARCH STARTED !!')
            gscv = GridSearchCV(pipeline, param, cv=tscv, refit=True)
            gscv.fit(train_x, train_y)
            best_estimetor = gscv.best_estimator_
            best_estimetors[pipeline_name] = best_estimetor
        joblib.dump(best_estimetors, f"{DATA_DIR}/best_estimetors.pkl")
    else:
        best_estimetors = joblib.load(f"{DATA_DIR}/best_estimetors.pkl")
    
    print(best_estimetors)
    meta_model = LogisticRegression() # meta_modelは線形モデル


    





if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executer:
        executer.submit(main()) # CPU4つ使っている。