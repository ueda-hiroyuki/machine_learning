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
    encorded_data = pd.concat([encorded_data, use, label], axis=1)
    
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)
    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    est = RandomForestClassifier(random_state=0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    selector = RFECV(estimator=est, step=0.05, n_jobs=-1, min_features_to_select=1, cv=skf, verbose=10)
    selector.fit(train_x, train_y)
    selected_columns = train_x.columns[selector.support_]
    selected_train_x = pd.DataFrame(selector.transform(train_x), columns=selected_columns)
    selected_test_x = test_x.loc[:, selected_columns]

    tr_x, val_x, tr_y, val_y = train_test_split(selected_train_x, train_y, test_size=0.2, stratify=train_y)

    estimators = [
        (
            "Gradient Boosting", 
            GradientBoostingClassifier(
                n_estimators=1000,
                max_depth=10,
                random_state=1
            )
        ),
        (
            "Random Forest", 
            RandomForestClassifier(
                n_estimators=1000,
                random_state=1,
                n_jobs=-1
            )
        ),
        (
            "LightGBM", 
            lgb.LGBMClassifier(
                n_estimators=1000,
                objective='multiclass',
                n_jobs=-1
            )
        ),
        (
            "Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty='l2',
                    max_iter=1000,
                    n_jobs=-1,
                )
            )
        )
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=lgb.LGBMClassifier(
            n_estimators=1000,
            objective='multiclass',
            n_jobs=-1
        ), 
        n_jobs=-1, 
        cv=skf,
        verbose=10
    )
    clf.fit(
        tr_x, 
        tr_y
        # eval_set=[(val_x, val_y)],
        # eval_metric='multi_logloss',
        # early_stopping_rounds=10
    )
    predictions = clf.predict_proba(selected_test_x)
    print(predictions, predictions.shape)

    #dataframe.to_csv('submission_sklearn_single_stacking_model.csv', index=False)


    # for pipe_name, pipeline in pipelines.items():
    #     print(f"########## START {pipe_name} Learning ##########")
    #     skf = StratifiedKFold(n_splits=5, shuffle=True)
    #     pipeline.fit(train_x, train_y) # learning
    #     results = cross_val_score(pipeline, train_x, train_y, scoring='f1_micro', cv=skf) # scoring with CV
    #     y_pred = pipeline.predict(test_x)
    #     print(y_pred)
    #     f1 = f1_score(y_pred, test_y, average="micro")
    #     print("###########################################")
    #     print(f'train auc: [{np.mean(results)}]')
    #     print(f'final {pipe_name} f1: [{f1}]')


if __name__ == "__main__":
    main()
