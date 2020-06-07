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


def get_important_components(train_x: t.Any, test_x: t.Any, pca_num: int):
    pca = PCA(n_components=pca_num).fit(train_x)
    pca_train_x = pca.transform(train_x)
    pca_test_x = pca.transform(test_x)
    return pca_train_x, pca_test_x


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
    return model


def main():
    train_pitching = pd.read_csv(TRAIN_PITCH_PATH, na_values='不明')
    train_player = pd.read_csv(TRAIN_PLAYER_PATH, na_values='不明')
    test_pitching = pd.read_csv(TEST_PITCH_PATH, na_values='不明')
    test_player = pd.read_csv(TEST_PLAYER_PATH, na_values='不明')
    se_player = pd.read_csv(EXTERNAL_1_PATH, encoding="cp932", na_values='-').rename(columns={"名前": "選手名", "打席": "打席数"}) 
    pa_player = pd.read_csv(EXTERNAL_2_PATH, na_values='-')
    pitcher_ability = pd.read_csv(EXTERNAL_3_PATH, na_values='-')
    player_ability = pd.read_csv(EXTERNAL_4_PATH, na_values='-')
    pitcher_result = pd.read_csv(EXTERNAL_7_PATH, na_values='-')

    pitcher_result = pitcher_result.rename(columns={"DeNA": "ＤｅＮＡ"}) 
    player_result = pd.concat([se_player, pa_player], axis=0)
    train_pitching["usage"] = "train"
    test_pitching["usage"] = "test"
    test_pitching["球種"] = 0

    pitching_data = pd.concat([train_pitching, test_pitching], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)
    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1)
    pitchers_data = player_data[player_data["位置"] == "投手"]

    merged_pitchers_data = pd.merge(
        pitchers_data, 
        pitcher_result, 
        how="left", 
        left_on=['チーム名','選手名', '年度'], 
        right_on=['チーム','選手名', '年度'],
    ).drop(['チーム'], axis=1)

    merged_player_data = pd.merge(
        player_data, 
        player_result, 
        how="left", 
        left_on=['チーム名','選手名', '年度', '背番号'], 
        right_on=['チーム','選手名', '年度', '背番号'],
    ).drop(['チーム'], axis=1)

    merged_data = pd.merge(
        pitching_data, 
        merged_pitchers_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1)

    for col in list(merged_data.columns):
        if merged_data[col].dtype != "object":
            merged_data[col] = merged_data[col].fillna(merged_data[col].mean())

    usage = merged_data.loc[:, "usage"]
    merged_data = merged_data.drop(["usage", "位置", "年度"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_data) 
    encorded_data = pd.concat([encorded_data, usage], axis=1)
 
    train = encorded_data[encorded_data["usage"] == "train"].drop("usage", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["usage"] == "test"].drop("usage", axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    pca_num = len(train_x.columns) - 1
    pca_train_x, pca_test_x = get_important_components(train_x, test_x, pca_num)  

    n_splits = 10
    num_class = 8
    best_params = get_best_params(pca_train_x, train_y, num_class) # 最適ハイパーパラメータの探索

    submission = np.zeros((len(pca_test_x),num_class))
    accs = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(kf.split(pca_train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset, best_params)
        
        y_pred = np.argmax(model.predict(val_x), axis=1) # 0~8の確率
        acc = accuracy_score(val_y, y_pred)
        accs[i] = acc
        print("#################################")
        print(f"accuracy: {acc}")
        print("#################################")
        y_preda = model.predict(pca_test_x, num_iteration=model.best_iteration) # 0~8の確率
        submission += y_preda

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print(accs)
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/my_submission22.csv", header=False)



if __name__ == "__main__":
    main()