import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
from glob import glob
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
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


def accuracy(preds, data, threshold=0.5):
    """精度 (Accuracy) を計算する関数"""
    weight = data.get_weight()
    true_label = data.get_label()
    pred_label = (preds > threshold).astype(int)
    acc = np.average(true_label == pred_label, weights=weight)
    return "accuracy", acc, True


def load_model(num, data_dir, algorithm_name):
    all_models = []
    for i in range(num):
        model = joblib.load(f"{data_dir}/{algorithm_name}_model_{i}.pkl")
        all_models.append((f"{algorithm_name}{i}", model))
    return all_models


def train_by_randomforest(
    train_x, train_y, kfold, best_params=None, algorithm_name=None
):
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        model = RandomForestClassifier(
            bootstrap=True,
            criterion="gini",
            max_depth=7,
            max_features="auto",
            min_impurity_split=1e-07,
            min_samples_leaf=10,
            n_estimators=100,
            n_jobs=4,
            verbose=10,
        )
        model.fit(tr_x, tr_y)

        y_pred = model.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)
        models.append(model)
        acc_results.append(accuracy)

    return models, acc_results


def train_by_lightgbm(train_x, train_y, kfold, best_params=None, algorithm_name=None):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": {"binary_logloss"},
        "num_leaves": 50,
        "min_data_in_leaf": 100,
        "learning_rate": 0.1,
        "feature_fraction": 0.7,
        "is_unbalance": True,
    }
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y, reference=tr_set)

        evals_result = {}
        model = lgb.train(
            params=params,
            train_set=tr_set,
            valid_sets=[val_set, tr_set],
            valid_names=["eval", "train"],
            num_boost_round=1000,
            early_stopping_rounds=100,
            verbose_eval=1,
            evals_result=evals_result,
            feval=accuracy,
        )

        models.append(model)
        acc_results.append(max(evals_result["eval"]["accuracy"]))

    return models, acc_results


def train_by_catboost(train_x, train_y, kfold, best_params=None, algorithm_name=None):
    categorical_columns = [x for x in train_x.columns if train_x[x].dtype == "object"]
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": {"binary_logloss"},
        "num_leaves": 50,
        "min_data_in_leaf": 100,
        "learning_rate": 0.1,
        "feature_fraction": 0.7,
        "is_unbalance": True,
    }
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        model = CatBoostClassifier(
            iterations=2000,
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

    y = train_data.loc[:, "y"]
    X = train_data.drop(["y", "id"], axis=1)
    test_x = test_data.drop(["y", "id"], axis=1)

    # aaa = train_test_split(X, y, test_size=0.2, stratify=y)
    lgb_models = load_model(num=5, data_dir=DATA_DIR, algorithm_name="lightgbm")
    cb_models = load_model(num=5, data_dir=DATA_DIR, algorithm_name="catboost")
    rf_models = load_model(num=5, data_dir=DATA_DIR, algorithm_name="randomforest")
    all_models = [*lgb_models, *cb_models, *rf_models]

    meta_model = LogisticRegression(
        penalty="l2",
        C=0.01,
        max_iter=200,
        verbose=10,
        n_jobs=4,
    )

    clf = StackingClassifier(
        estimators=all_models,
        final_estimator=meta_model,
    )
    print(all_models)
    clf.fit(X, y)
    y_pred = clf.predict(test_x)

    print(y_pred)


if __name__ == "__main__":
    run_all()