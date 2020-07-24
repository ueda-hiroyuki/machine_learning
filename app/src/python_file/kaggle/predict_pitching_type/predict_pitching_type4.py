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
from imblearn.over_sampling import SMOTE
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

NUM_CLASS = 8

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

# pseudo_labeling
def gen_pseudo_data(train_x, train_y, test_x, n_splits, num_class, best_params):
    pseudo = np.zeros((len(test_x),num_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(skf.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        # model, evals_result = get_model(tr_x, tr_y, val_x, val_y, num_class, best_params)
        # y_preda = model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率
        model = get_balanced_weight_model(tr_x, tr_y, val_x, val_y, num_class, best_params)
        y_preda = model.predict_proba(test_x, num_iteration=model.best_iteration_) # 0~8の確率
        pseudo += pd.DataFrame(y_preda)

    pseudo_data = pseudo / n_splits
    mask = pseudo_data.max(axis=1) > 0.8
    pseudo_label = pseudo_data[mask].idxmax(axis=1)
    pseudo_idx = list(pseudo_label.index)

    print(pseudo_label.value_counts())
    print("####################################")
    # print(test_x)
    # print(test_x.iloc[pseudo_idx])
    # print(pseudo_label.value_counts())

    pseudo_train = test_x.iloc[pseudo_idx]
    pseudo_train["球種"] = pseudo_label

    pseudo_train_x = pd.concat([train_x, pseudo_train.drop(["球種"], axis=1)], axis=0).reset_index(drop=True)
    pseudo_train_y = pd.concat([train_y, pseudo_train.loc[:, "球種"]], axis=0).reset_index(drop=True)

    # print(train_x, pseudo_train_x)
    # print(train_y.value_counts(), pseudo_train_y.value_counts())
    
    pseudo_train = pd.concat([pseudo_train_x, pseudo_train_y], axis=1).drop_duplicates(subset=pseudo_train_x.columns).reset_index(drop=True)
    # print((pseudo_train.duplicated().value_counts())
    # printpseudo_train)
    return pseudo_train.drop(["球種"], axis=1), pseudo_train.loc[:, "球種"]



def gen_resampled_data(train_x, train_y):
    label_counts = train_y.value_counts()
    sm = SMOTE(
        ratio={
            0:sum(train_y==0)*int(round(label_counts[0]/label_counts[0])), 
            1:sum(train_y==1)*int(round(label_counts[0]/label_counts[1])),
            2:sum(train_y==2)*int(round(label_counts[0]/label_counts[2])),
            3:sum(train_y==3)*int(round(label_counts[0]/label_counts[3])),
            4:sum(train_y==4)*int(round(label_counts[0]/label_counts[4])),
            5:sum(train_y==5)*int(round(label_counts[0]/label_counts[5])),
            6:sum(train_y==6)*int(round(label_counts[0]/label_counts[6])),
            7:sum(train_y==7)*int(round(label_counts[0]/label_counts[7]))
        }
    )
    train_x_resampled, train_y_resampled = sm.fit_sample(train_x, train_y)
    train_x_resampled = pd.DataFrame(train_x_resampled, columns=train_x.columns)
    train_y_resampled = pd.Series(train_y_resampled, name="球種")

    return train_x_resampled, train_y_resampled


def get_model(tr_x, tr_y, val_x, val_y, num_class, best_params):
    train_set = lgb.Dataset(tr_x, tr_y, free_raw_data=False)
    valid_set = lgb.Dataset(val_x, val_y, reference=train_set, free_raw_data=False)
    evals_result = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate' : 0.05,
        'num_class': num_class,
        **best_params
    }
    model = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set, train_set],
        num_boost_round=100,
        # learning_rates=lambda iter: 0.1 * (0.99 ** iter),
        # callbacks=[lgb.reset_parameter(learning_rate=[0.1] * 1000)],
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=evals_result,
    )

    importance = pd.DataFrame(model.feature_importance(), index=tr_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance.head(50))
    return model, evals_result

def get_balanced_weight_model(tr_x, tr_y, val_x, val_y, num_class, best_params):
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type='gbdt',
        n_estimators=10000, 
        learning_rate=0.1,
        class_weight='balanced',
        min_data_in_leaf=200,
        random_state=1,
        n_jobs=4, 
        num_leaves=12,
        feature_fraction = 0.75,
        lambda_l1 = 5.96,
        lambda_l2 = 1.1,
        bagging_fraction= 0.89,
    )
    model = gbm.fit(
        tr_x, 
        tr_y,
        eval_set=[(tr_x, tr_y), (val_x, val_y)],
        early_stopping_rounds=20,
        verbose=20
    )
    importance = pd.DataFrame(model.feature_importances_, index=tr_x.columns, columns=['importance']).sort_values('importance', ascending=[False])
    # print(importance.head(50))
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
    merged = merged.drop(["use", "位置", "年度", "投手名"], axis=1)

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

    # train_x_resampled, train_y_resampled = gen_resampled_data(train_x, train_y)
    train_x_resampled, train_y_resampled = train_x, train_y

    n_splits = 3
    num_class = 8
    best_params = {
        'lambda_l1': 5.96,
        'lambda_l2': 1.1,
        'num_leaves': 12,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.89,
        'bagging_freq': 7,
        'min_data_in_leaf': 200
    }

    pseudo_counts = 3
    for i in range(pseudo_counts):
        if i == 0:
            pseudo_train_x, pseudo_train_y = gen_pseudo_data(train_x_resampled, train_y_resampled, test_x, n_splits, num_class, best_params)
        else:
            pseudo_train_x, pseudo_train_y = gen_pseudo_data(pseudo_train_x, pseudo_train_y, test_x, n_splits, num_class, best_params)
    
    submission = np.zeros((len(test_x),num_class))
    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(skf.split(pseudo_train_x, pseudo_train_y)):
        tr_x = pseudo_train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = pseudo_train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = pseudo_train_x.iloc[val_idx].reset_index(drop=True)
        val_y = pseudo_train_y.iloc[val_idx].reset_index(drop=True)

        # model, evals_result = get_model(tr_x, tr_y, val_x, val_y, num_class, best_params)
        # # 学習曲線の描画
        # fig = lgb.plot_metric(evals_result, metric="multi_logloss")
        # plt.savefig(f"{DATA_DIR}/learning_curve_{i}.png")
        # y_preda = model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率

        model = get_balanced_weight_model(tr_x, tr_y, val_x, val_y, num_class, best_params)
        y_preda = model.predict_proba(test_x, num_iteration=model.best_iteration_) # 0~8の確率
        
        submission += y_preda

    submission_df = pd.DataFrame(submission)/n_splits
    submission_df.to_csv(f"{DATA_DIR}/my_submission41.csv", header=False)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print("#################################")


    


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executer:
        executer.submit(main()) # CPU4つ使っている。