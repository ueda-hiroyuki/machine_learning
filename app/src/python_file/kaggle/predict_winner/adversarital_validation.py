import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
from glob import glob
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from python_file.kaggle.predict_winner.common import Common
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoost, Pool, CatBoostClassifier
from sklearn.ensemble import StackingClassifier


DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
BUKI_PATH = f"{DATA_DIR}/weapon.csv"

REMOVAL_COLS = ["lobby", "game-ver", "period"]

WEAPON_MAP = {
    "heroblaster_replica": "hotblaster",
    "herobrush_replica": "hokusai",
    "herocharger_replica": "splatcharger",
    "heromaneuver_replica": "maneuver",
    "heroroller_replica": "splatroller",
    "heroshelter_replica": "parashelter",
    "heroshooter_replica": "sshooter",
    "heroslosher_replica": "bucketslosher",
    "herospinner_replica": "splatspinner",
    "octoshooter_replica": "sshooter",
}

RANK_MAP = {
    "c-": 1,
    "c": 2,
    "c+": 3,
    "b-": 4,
    "b": 5,
    "b+": 6,
    "a-": 7,
    "a": 8,
    "a+": 9,
    "s-": 10,
    "s": 11,
    "s+": 12,
    "x": 13,
}

cm = Common()


def train(train_x, train_y, kfold, best_params=None, algorithm_name=None):
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            use_best_model=True,
            # one_hot_max_size=1000,
            eval_metric="Accuracy",
        )
        model.fit(
            tr_x,
            tr_y,
            # cat_features=categorical_columns,
            eval_set=(val_x, val_y),
            plot=True,
        )

        y_pred = model.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)

        models.append(model)
        acc_results.append(accuracy)

    return models, acc_results


def predict(model, test_x, threshold):
    y_pred = model.predict(test_x)
    pred = [0 if i < threshold else 1 for i in y_pred]
    return pd.Series(pred)


def detect_adversarital_idx(train_x, test_x):
    """
    Adversarital_Validation: テストデータの分布に近いデータを抽出する
    ⇒ 学習用と検証用のデータ分布に差があるとき使用し、よりテストデータにフィットしたモデルを作成することができる。
    """

    # 学習用とテスト用データの疑似ラベルを作成する。
    train_z = pd.Series(np.zeros(len(train_x)), name="y")  # 片方は全て 0
    test_z = pd.Series(np.ones(len(test_x)), name="y")  # もう片方は全て 1

    adv_data = pd.concat([train_x, test_x], axis=0)
    adv_label = pd.concat([train_z, test_z], axis=0)

    clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        eval_metric="Accuracy",
    )

    # 分類して確率を計算する
    pred_proba = cross_val_predict(
        clf, adv_data, adv_label, cv=3, method="predict_proba"
    )
    proba_df = pd.DataFrame(pred_proba, columns=["train", "test"])

    adv_train = proba_df[: len(train_x)]  # train側の予測確率を分離
    adv_test = proba_df[len(train_x) :].reset_index(drop=True)  # test側の予測確率を分離
    pred_train_idx = adv_train[
        adv_train["test"] > 0.5
    ].index  # trainデータの中でテストっぽいデータだと判断された行のindex

    pred_train_idx = (
        adv_train.loc[:, "test"].sort_values(ascending=False).head(20000).index
    )
    return pred_train_idx


def run_all():
    train_raw_data = pd.read_csv(TRAIN_PATH)
    test_raw_data = pd.read_csv(TEST_PATH)
    buki_raw_data = pd.read_csv(BUKI_PATH)

    train_raw_data = train_raw_data.replace(WEAPON_MAP)
    test_raw_data = test_raw_data.replace(WEAPON_MAP)
    buki_raw_data = buki_raw_data.replace(WEAPON_MAP)

    test_raw_data["y"] = 0
    train_raw_data["usage"] = 0  # for train
    test_raw_data["usage"] = 1  # for test

    X = cm.make_input_output(train_raw_data, with_y=False)
    train_data = pd.concat([train_raw_data, X], axis=1)

    # 武器ごとの勝率計算
    win_rate_df = []
    for buki in buki_raw_data.key.unique():
        rate, count = cm.win_rate(buki, train_data)
        win_rate_df.append([buki, rate, count])
    win_rate_df = pd.DataFrame(win_rate_df, columns=["buki", "win_rate", "count"])
    win_rate_df = win_rate_df.sort_values(by="win_rate", ascending=False)
    buki_ja_dict = buki_raw_data[["key", "reskin"]].set_index("key").to_dict()["reskin"]
    win_rate_df.buki = win_rate_df.buki.map(buki_ja_dict)
    win_rate_dict = (
        win_rate_df.reset_index(drop=True)
        .loc[:, ["buki", "win_rate"]]
        .set_index("buki")
        .to_dict()
    )[
        "win_rate"
    ]  # 全体の勝率dict(武器名: 勝率)

    # モード毎の武器勝率計算
    win_rate_mode_df = pd.DataFrame()
    for mode in train_data["mode"].unique():
        win_rate_df = []
        train_mode = train_data[train_data["mode"] == mode]
        for buki in buki_raw_data.key.unique():
            # print(buki, win_rate(buki))
            rate, count = cm.win_rate(buki, train_mode)
            win_rate_df.append([buki, rate, count])
        win_rate_df = pd.DataFrame(
            win_rate_df, columns=[mode + "_buki", mode + "_win_rate", mode + "_count"]
        )
        win_rate_df = win_rate_df.sort_values(by=mode + "_win_rate", ascending=False)
        buki_ja_dict = (
            buki_raw_data[["key", "reskin"]].set_index("key").to_dict()["reskin"]
        )
        win_rate_df[mode + "_buki"] = win_rate_df[mode + "_buki"].map(buki_ja_dict)
        win_rate_df = win_rate_df.reset_index(drop=True)
        win_rate_mode_df = pd.concat([win_rate_mode_df, win_rate_df], axis=1)

    # 武器と射程距離の関係分類計算
    buki_range_distance_dict = (
        buki_raw_data.loc[:, ["category2", "key"]]
        .set_index("key")
        .to_dict()["category2"]
    )

    train_data, test_data = cm.make_feature(train_raw_data, test_raw_data)
    raw_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    raw_data = raw_data.fillna(raw_data.mode().iloc[0]).drop(REMOVAL_COLS, axis=1)
    for name in raw_data.columns:
        if "rank" in name:
            raw_data[name] = raw_data[name].map(RANK_MAP)

    data = cm.calc_team_level_avg(raw_data)
    data = cm.calc_team_rank_avg(data)
    data = cm.add_count_range_distance(data, buki_range_distance_dict)
    data = cm.calc_weapons_win_rate_avg(data, win_rate_dict)
    data = cm.calc_weapons_win_rate_avg_per_mode(data, win_rate_mode_df)

    categorical_columns = [x for x in data.columns if data[x].dtype == "object"]

    ce_oe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown="impute")
    encoded_data = ce_oe.fit_transform(data)

    train_data = (
        encoded_data[encoded_data["usage"] == 0]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )
    test_data = (
        encoded_data[encoded_data["usage"] == 1]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )

    ids = test_data.loc[:, "id"]

    train_y = train_data.loc[:, "y"]
    train_x = train_data.drop(["y", "id"], axis=1)
    test_x = test_data.drop(["y", "id"], axis=1)

    # adversarital_validationの実施(分布がtestデータに近いtrainデータを抽出)
    pred_train_idx = detect_adversarital_idx(train_x, test_x)  # train内のtestっぽいデータ
    selected_train_x = train_x.iloc[pred_train_idx]
    selected_train_y = train_y.iloc[pred_train_idx]

    # 学習
    n_splits = 5
    algorithm_name = "catboost"
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    models, acc_results = train(train_x, train_y, kfold)

    add_data = np.zeros((len(test_x), 2))
    for i, model in enumerate(models):
        y_pred = model.predict_proba(test_x)
        add_data += y_pred
    add_data = pd.DataFrame(add_data / len(models))
    pseudo_label = add_data[(add_data[0] > 0.8) | (add_data[1] > 0.8)].idxmax(
        axis=1
    )  # 予測確率の高い行の疑似正解ラベルを取得する

    pseudo_data = pd.concat(
        [test_x.iloc[pseudo_label.index], pseudo_label], axis=1
    ).rename(columns={0: "y"})

    new_train_x = pd.concat(
        [train_x, pseudo_data.drop("y", axis=1)], axis=0
    ).reset_index(drop=True)
    new_train_y = pd.concat([train_y, pseudo_label], axis=0).reset_index(drop=True)

    # pseudo_labeling後の再学習
    models, acc_results = train(
        train_x=new_train_x,
        train_y=new_train_y,
        kfold=kfold,
        algorithm_name=algorithm_name,
    )

    # 評価
    threshold = 0.5
    y_preds = []
    for i, model in enumerate(models):
        y_pred = predict(model, test_x, threshold)
        y_preds.append(y_pred)

    # 提出用ファイル成型
    winner_pred = pd.concat(y_preds, axis=1).mode(axis=1).rename(columns={0: "y"})
    submission = pd.concat([ids, winner_pred], axis=1)

    print(submission)
    print("######################################")
    print(f"accuracy avg = {sum(acc_results) / len(acc_results)}")
    print("######################################")
    submission.to_csv(f"{DATA_DIR}/submission41.csv", index=False)


if __name__ == "__main__":
    run_all()