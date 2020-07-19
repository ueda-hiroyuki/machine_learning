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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, TimeSeriesSplit, GridSearchCV
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


PITCH_REMOVAL_COLUMNS = ["時刻", "データ内連番", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

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
    "投手試合内対戦打者数",
    "昨年度_投手_防御率",
    "イニング",
    "昨年度_投手_敗北",
    "昨年度_投手_被安打",
    "昨年度_投手_奪三振",
    "昨年度_投手_打者",
    "昨年度_投手_投球回",
    "昨年度_投手_自責点",
    "昨年度_投手_失点",
    "昨年度_投手_DIPS",
    "昨年度_打者_OPS",
    "昨年度_打者_打数",
    "昨年度_打者_打席数",
    "昨年度_打者_打率",
    "昨年度_打者_三振",
    "昨年度_打者_試合",
    "昨年度_打者_RC27",
    "昨年度_打者_長打率",
    "昨年度_打者_本塁打",
    "昨年度_打者_四球",
]


TEAM_ID_MAP = {
    1: "巨人",
    2: "ヤクルト",
    3: "ＤｅＮＡ",
    4: "中日",
    5: "阪神",
    6: "広島",
    7: "西武",
    8: "日本ハム",
    9: "ロッテ",
    10: "オリックス",
    11: "ソフトバンク",
    12: "楽天"
}


NUM_CLASS = 8


def preprocessing(df):
    #df['走者'] = np.where(df["プレイ前走者状況"] == "___", 0, 1) # プレイ前ランナーがいるかいないか。
    df['BMI'] = (df["体重"]/(df["身長"]/100)**2) # 身長体重をBMIに変換
    df = df.drop(["体重", "身長"], axis=1)
    return df


def get_selected_columns(train_x, train_y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    est = lgb.LGBMClassifier()
    selector = RFECV(estimator=est, step=0.05, n_jobs=2, min_features_to_select=round(len(train_x)*0.8), cv=skf, verbose=10)
    selector.fit(train_x, train_y)
    selected_columns = train_x.columns[selector.support_]
    return selected_columns


def objective(X, y, trial):
    """最適化する目的関数"""
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, random_state=1)
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt',
        n_jobs = 2,
        n_estimators=1000,
    )
    # RFE で取り出す特徴量の数を最適化する
    n_features_to_select = trial.suggest_int('n_features_to_select', 1, len(list(tr_x.columns))),
    rfe = RFE(estimator=gbm, n_features_to_select=n_features_to_select)
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
    f1 = f1_score(val_y, y_pred, average="micro")
    return f1


def get_best_params(train_x, train_y, valid_x, valid_y) -> t.Any:
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        early_stopping_rounds=50,
        eval_metric='multi_logloss',
    )
    params = {
        'max_depth' : [5, 10],
        'num_leaves' : [10, 50, 100],
        'min_data_in_leaf': [10, 100, 1000],
    }
    clf = GridSearchCV(
        estimator = clf,
        param_grid = params,
    )
    clf.fit(
        train_x,
        train_y,
        eval_set = [[valid_x, valid_y]]
    )
    best_params = clf.best_params_

    return best_params

def get_model(train_x, train_y, valid_x, valid_y, num_class, best_params) -> t.Any:
    best_params = {
        'lambda_l1': 5.96,
        'lambda_l2': 1.1,
        'num_leaves': 12,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.89,
        'bagging_freq': 7,
        #'min_child_sample': 100
    }
    # 学習用データセット
    train_set = lgb.Dataset(train_x, train_y, free_raw_data=False)
    # 評価用データセット
    valid_set = lgb.Dataset(valid_x, valid_y, free_raw_data=False)
    evals_result = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_class': num_class,
        **best_params
    }
    model = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set, train_set],
        num_boost_round=1000,
        verbose_eval=10,
        # learning_rates=lambda iter: 0.1 * (0.99 ** iter),
        callbacks=[lgb.reset_parameter(learning_rates=[0.2] * 400 + [0.1] * 400 + [0.05] * 200)],
        evals_result=evals_result,
    )
    importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance.head(50))
    return model, evals_result


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, parse_dates=["日付"])
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitching = pd.read_csv(TEST_PITCH_PATH, parse_dates=["日付"])
    test_player = pd.read_csv(TEST_PLAYER_PATH)

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

    # train_pitching = train_pitching.head(100) # メモリ節約のため
    # test_pitching = test_pitching.head(100) # メモリ節約のため

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

    # データセットと前年度投手成績データをmergeする
    pitchers_results = pitchers_results.replace({'チーム': {"DeNA": "ＤｅＮＡ"}})
    pitchers_results = pitchers_results.rename(columns={"チーム": "投手チーム名"})
    pitchers_results["年度"] = pitchers_results["年度"] + 1
    pitchers_results = pitchers_results[pitchers_results["年度"] >= 2017]
    replasce_cols = list(set(pitchers_results.columns) - set(["年度", "投手チーム名", "選手名"]))
    for col in replasce_cols:
        pitchers_results = pitchers_results.rename(columns={col: f'昨年度_投手_{col}'})

    merged = pd.merge(
        merged,
        pitchers_results,
        how="left",
        left_on=['年度','投手名', "投手チーム名"],
        right_on=['年度','選手名', "投手チーム名"]
    ).drop(['選手名'], axis=1).fillna(0) # 今年から登板の投手のデータは0で置換する。

    # # データセットと前年度打者成績データをmergeする
    # batters_results = batters_results.replace({'チーム': {"DeNA": "ＤｅＮＡ"}})
    # batters_results = batters_results.rename(columns={"チーム": "打者チーム名", "選手名": "打者名"})
    # batters_results["年度"] = batters_results["年度"] + 1
    # batters_results = batters_results[batters_results["年度"] >= 2017].reset_index(drop=True)
    # replasce_cols = list(set(batters_results.columns) - set(["年度", "打者チーム名", "打者名"]))
    # for col in replasce_cols:
    #     batters_results = batters_results.rename(columns={col: f'昨年度_打者_{col}'})

    # playersID = player_data[['年度', "選手ID", "選手名", "チーム名"]]
    # playersID = playersID.rename(columns={"チーム名": "打者チーム名", "選手名": "打者名"})
    # merged = pd.merge(
    #     merged,
    #     playersID,
    #     how="left",
    #     left_on=['年度', "打者ID"],
    #     right_on=['年度', "選手ID"]
    # ).drop(['選手ID'], axis=1)

    # merged = pd.merge(
    #     merged,
    #     batters_results,
    #     how="left",
    #     on=['年度', '打者名', "打者チーム名"]
    # ).drop(["打者チーム名", "投手チーム名", "打者名", "投手名"], axis=1).fillna(0) # 昨年度のデータがない打者のデータは0で置換する。

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
    encorded = encorded.drop(FINAL_REMOVAL_COLUMNS, axis=1)

    # cf.check_corr(encorded, "predict_pitching_type")


    param_train = encorded[(encorded["use"] == "train") & (encorded["日付"] < "2017-9-1")].drop(["use"], axis=1).reset_index(drop=True)
    param_valid = encorded[(encorded["use"] == "train") & (encorded["日付"] >= "2017-9-1")].drop(["use"], axis=1).reset_index(drop=True)
    # test = encorded[(encorded["use"] == "test")].drop(["use","日付"], axis=1).reset_index(drop=True)

    train = encorded[(encorded["use"] == "train")].drop(["use","日付"], axis=1).reset_index(drop=True)
    test = encorded[(encorded["use"] == "test")].drop(["use","日付"], axis=1).reset_index(drop=True)

    param_train_x = param_train.drop(["球種", "日付"], axis=1)
    param_train_y = param_train.loc[:,"球種"]
    param_valid_x = param_valid.drop(["球種", "日付"], axis=1)
    param_valid_y = param_valid.loc[:,"球種"]

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    # GridSearchによる最適パラメータの探索
    if not os.path.isfile(f"{DATA_DIR}/best_params.pkl"):
        best_params = get_best_params(param_train_x, param_train_y, param_valid_x, param_valid_y)
        joblib.dump(best_params, f"{DATA_DIR}/best_params.pkl")
    else:
        best_params = joblib.load(f"{DATA_DIR}/best_params.pkl")

    n_splits = 5
    submission = np.zeros((len(test_x),NUM_CLASS))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    alphas = [1.03, 1.02, 1.01, 0.99, 0.98, 0.97]
    weights = [1/len(alphas)] * len(alphas)
    for  icount, (alpha, weight) in enumerate(zip(alphas, weights)):
        preds = np.zeros((len(test_x),NUM_CLASS))
        for i, (tr_idx, val_idx) in enumerate(tscv.split(train_x)):
            tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
            tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
            val_x = train_x.iloc[val_idx].reset_index(drop=True)
            val_y = train_y.iloc[val_idx].reset_index(drop=True)
            model, evals_result = get_model(tr_x, tr_y, val_x, val_y, NUM_CLASS, best_params)
            
            # 学習曲線の描画
            fig = lgb.plot_metric(evals_result, metric="multi_logloss")
            plt.savefig(f"{DATA_DIR}/learning_curve_{icount}_{i}.png")
            y_preda = alpha * model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率
            preds += y_preda

        submission += preds * weight

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################################")
    print(submission_df)
    print("#################################")

    submission_df.to_csv(f"{DATA_DIR}/my_submission30.csv", header=False)





if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executer:
        executer.submit(main()) # CPU4つ使っている。