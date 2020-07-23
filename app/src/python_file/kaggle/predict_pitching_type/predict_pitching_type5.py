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
        n_estimators=100,
        random_state=42
    )

    # 分類して確率を計算する
    z_pred_proba = cross_val_predict(
        clf, 
        adv_data,
        adv_label,
        cv=5, 
        method='predict_proba'
    )
    preda_df = pd.DataFrame(z_pred_proba, columns=["train", "test"])
    adv_train = preda_df[:len(train_x)] # train側の予測確率を分離
    adv_test = preda_df[len(train_x):] # test側の予測確率を分離
    pred_train_idx = adv_train[adv_train["test"] > 0.6].index # trainデータの中でテストっぽいデータだと判断された行のindex
    return pred_train_idx

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


def get_model(tr_dataset: t.Any, val_dataset: t.Any, num_class: int, best_params: t.Dict[str, t.Any], cols: t.Sequence[str]) -> t.Any:
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
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        num_boost_round=1000,
        # learning_rates=lambda iter: 0.1 * (0.99 ** iter),
        # callbacks=[lgb.reset_parameter(learning_rate=[0.1] * 1000)],
        early_stopping_rounds=50,
        verbose_eval=10,
        evals_result=evals_result,
    )

    importance = pd.DataFrame(model.feature_importance(), index=cols, columns=['importance']).sort_values('importance', ascending=[False])
    print(importance.head(50))
    return model, evals_result


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
        print("###############################")
        print(f"{i}th generate adversarial validation")
        print("###############################")
        if i == 0:
            pred_train_idx = gen_adversarital_data(train_x, test_x)
            pred_train_x = train_x.iloc[pred_train_idx].reset_index(drop=True)
            pred_train_y = train_y.iloc[pred_train_idx].reset_index(drop=True)
        else:
            pred_train_idx = gen_adversarital_data(pred_train_x, test_x)
            pred_train_x = pred_train_x.iloc[pred_train_idx].reset_index(drop=True)
            pred_train_y = pred_train_y.iloc[pred_train_idx].reset_index(drop=True)
    # print("######################################")
    # print(pred_train_x, pred_train_y)
    # print(pred_train_y.value_counts())

    train_x_resampled, train_y_resampled = gen_resampled_data(pred_train_x, pred_train_y)
    print("######################################")
    print(train_x_resampled)
    print(train_y_resampled.value_counts())

    n_splits = 5
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

    submission = np.zeros((len(test_x),num_class))
    accs = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(skf.split(train_x_resampled, train_y_resampled)):
        tr_x = train_x_resampled.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y_resampled.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x_resampled.iloc[val_idx].reset_index(drop=True)
        val_y = train_y_resampled.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y, free_raw_data=False)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset, free_raw_data=False)
        model, evals_result = get_model(tr_dataset, val_dataset, num_class, best_params, train_x_resampled.columns)
        
        # 学習曲線の描画
        fig = lgb.plot_metric(evals_result, metric="multi_logloss")
        plt.savefig(f"{DATA_DIR}/learning_curve_{i}.png")

        y_pred = np.argmax(model.predict(val_x), axis=1) # 0~8の確率
        acc = accuracy_score(val_y, y_pred)
        accs[i] = acc
        print("#################################")
        print(f"accuracy: {acc}")
        print("#################################")
        y_preda = model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率
        submission += y_preda

    submission_df = pd.DataFrame(submission/n_splits)
    submission_df.to_csv(f"{DATA_DIR}/my_submission36.csv", header=False)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print(accs)
    print("#################################")


    




        



if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1) as executer:
        executer.submit(main()) # CPU4つ使っている。