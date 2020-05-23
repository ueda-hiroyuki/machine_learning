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
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
from sklearn.ensemble import StackingClassifier
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


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    y_true = data.get_label()
    y_preds = np.reshape(preds, [len(y_true), 8], order='F')
    y_pred = np.argmax(y_preds, axis=1)
    metric = np.mean(y_true == y_pred)
    return 'accuracy', metric, True

def get_best_params(train_x: t.Any, train_y: t.Any, num_class: int) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y)
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
        verbose_eval=50,
        best_params=best_params,
        tuning_history=tuning_history
    )
    return best_params

def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'num_class': NUM_CLASS,
        **params
    }
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=10000,
        valid_names=['eval','train'],
        evals_result=evals_result,
        verbose_eval=50,
        feval=accuracy,
    )
    return model, evals_result

def objective(X, y, trial):
    """最適化する目的関数"""
    print("Run objective")
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, random_state=1)
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        #n_jobs = 4,
        n_estimators=10000,
    )
    # RFE で取り出す特徴量の数を最適化する
    n_features_to_select = trial.suggest_int('n_features_to_select', 1, len(list(tr_x.columns))),
    rfe = RFE(estimator=gbm, n_features_to_select=n_features_to_select, verbose=50, step=3)
    rfe.fit(tr_x, tr_y)
    selected_cols = list(tr_x.columns[rfe.support_])
    
    tr_x_selected = tr_x.loc[:, selected_cols]
    val_x_selected = val_x.loc[:, selected_cols]
    gbm.fit(
        tr_x_selected, 
        tr_y,
        eval_set=[(val_x_selected, val_y)],
        early_stopping_rounds=20
    )
    y_pred = gbm.predict(val_x_selected)
    kappa = cohen_kappa_score(val_y, y_pred)
    return kappa

def get_important_features(train_x: t.Any, train_y: t.Any, best_feature_count: int):
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        #n_jobs = 4,
    )
    print("Start RFE")
    selector = RFE(gbm, n_features_to_select=best_feature_count, verbose=50)
    selector.fit(train_x, train_y) # 学習データを渡す
    print("Finished RFE")
    selected_train_x = pd.DataFrame(selector.transform(train_x), columns=train_x.columns[selector.support_])
    return selected_train_x, train_y


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

    labal = merged_data.loc[:, "球種"]
    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use", "位置", "年度", "球種"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_data) 
    

    f = partial(objective, encorded_data, labal) # 目的関数に引数を固定しておく
    study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する
    study.optimize(f, n_trials=10) # 試行回数を決定する
    print('params:', study.best_params)# 発見したパラメータを出力する
    best_feature_count = study.best_params['n_features_to_select']
    selected_data, selected_label = get_important_features(encorded_data, labal, best_feature_count)
    selected_data = pd.concat([selected_data, use, selected_label], axis=1)

    train = selected_data[selected_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = selected_data[selected_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    best_params = get_best_params(train_x, train_y, NUM_CLASS)
    print(best_params)
 
    n_splits = 10
    submission = np.zeros((len(test_x), NUM_CLASS))
    kappas = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    estimators = [
        (
            'lgb', 
            lgb.LGBMClassifier(**best_params, verbose=10)
        ),
        (
            'lr', 
            LogisticRegression(
                penalty='l2',
                max_iter=1000,
            )
        ),
    ]
    fin_est = LogisticRegression(penalty='l2',max_iter=10)
    clf = StackingClassifier(estimators=estimators, final_estimator=fin_est, cv=skf, verbose=10)
    clf.fit(train_x, train_y)
    predictions = clf.predict_proba(test_x)
    submission = pd.DataFrame(predictions)
    print(submission)
    submission.to_csv(f"{DATA_DIR}/my_submission20.csv", header=False)


if __name__ == "__main__":
    main()
