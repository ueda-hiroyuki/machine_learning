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
from sklearn.neural_network import MLPClassifier
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

# ハンドラーの定義
def handler(function, *arguments):
    return function(*arguments)


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
    train_x,
    train_y,
    kfold,
    best_params=None,
    algorithm_name=None,
):
    models = []
    preds = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        model = RandomForestClassifier(
            bootstrap=True,
            criterion="gini",
            max_depth=7,
            max_features="auto",
            min_impurity_split=1e-07,
            min_samples_leaf=10,
            n_estimators=200,
            # n_estimators=1,
            n_jobs=4,
            verbose=10,
        )
        model.fit(tr_x, tr_y)
        y_pred = model.predict(val_x)
        pred = [0 if i < 0.5 else 1 for i in y_pred]
        models.append(model)
        preds.append(pd.DataFrame(pred, list(val_x.index)))
    preds = pd.concat(preds, axis=0).sort_index()
    return models, preds


def train_by_neuralnet(
    train_x,
    train_y,
    kfold,
    best_params=None,
    algorithm_name=None,
):
    models = []
    preds = []
    model = MLPClassifier(
        activation="relu",
        batch_size="auto",
        early_stopping=True,
        hidden_layer_sizes=(100, 100, 100, 100, 100),
        learning_rate_init=0.1,
        # max_iter=1,
        max_iter=200,
        momentum=0.9,
        n_iter_no_change=10,
        random_state=1,
        shuffle=True,
        solver="sgd",
        verbose=10,
    )
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        model.fit(tr_x, tr_y)
        y_pred = model.predict(val_x)
        pred = [0 if i < 0.5 else 1 for i in y_pred]
        models.append(model)
        preds.append(pd.DataFrame(pred, list(val_x.index)))
    preds = pd.concat(preds, axis=0).sort_index()
    return models, preds


def train_by_lightgbm(
    train_x,
    train_y,
    kfold,
    best_params=None,
    algorithm_name=None,
):
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
    preds = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        tr_set = lgb.Dataset(tr_x, tr_y)
        val_set = lgb.Dataset(val_x, val_y, reference=tr_set)

        evals_result = {}
        model = lgb.train(
            params=params,
            train_set=tr_set,
            valid_sets=[val_set, tr_set],
            valid_names=["eval", "train"],
            # num_boost_round=1,
            num_boost_round=1000,
            early_stopping_rounds=100,
            verbose_eval=1,
            evals_result=evals_result,
            feval=accuracy,
        )

        y_pred = model.predict(val_x)
        pred = [0 if i < 0.5 else 1 for i in y_pred]
        models.append(model)
        preds.append(pd.DataFrame(pred, list(val_x.index)))
    preds = pd.concat(preds, axis=0).sort_index()
    return models, preds


def train_by_catboost(
    train_x,
    train_y,
    kfold,
    best_params=None,
    algorithm_name=None,
):
    models = []
    preds = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx]
        val_y = train_y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=1000,
            # iterations=1,
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
        pred = [0 if i < 0.5 else 1 for i in y_pred]
        models.append(model)
        preds.append(pd.DataFrame(pred, list(val_x.index)))
    preds = pd.concat(preds, axis=0).sort_index()
    return models, preds


def gen_pseudo_label(models, X, y, test_x, flg=False):
    add_data = np.zeros((len(test_x), 2))
    for i, model in enumerate(models):
        if flg:
            y_pred = model.predict(test_x)
            y_pred = [[1 - i, i] for i in y_pred]
        else:
            y_pred = model.predict_proba(test_x)
        add_data += y_pred
    add_data = pd.DataFrame(add_data / len(models))
    pseudo_label = add_data[(add_data[0] > 0.8) | (add_data[1] > 0.8)].idxmax(
        axis=1
    )  # 予測確率の高い行の疑似正解ラベルを取得する

    pseudo_data = pd.concat(
        [test_x.iloc[pseudo_label.index], pseudo_label], axis=1
    ).rename(columns={0: "y"})

    new_train_x = pd.concat([X, pseudo_data.drop("y", axis=1)], axis=0).reset_index(
        drop=True
    )
    new_train_y = pd.concat([y, pseudo_label], axis=0).reset_index(drop=True)
    return new_train_x, new_train_y


def run_train_and_stacking(X, y, test_x):
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # 第1層目の学習モデル(cv)、CVのvalidationデータに対する予測値(metaモデルの学習データ)
    cat_base_models, cat_base_preds_valid = train_by_catboost(X, y, kfold)
    lgb_base_models, lgb_base_preds_valid = train_by_lightgbm(X, y, kfold)
    rf_base_models, rf_base_preds_valid = train_by_randomforest(X, y, kfold)
    nn_base_models, nn_base_preds_valid = train_by_neuralnet(X, y, kfold)

    # pseudo_labeling
    cat_new_X, cat_new_y = gen_pseudo_label(cat_base_models, X, y, test_x)
    lgb_new_X, lgb_new_y = gen_pseudo_label(lgb_base_models, X, y, test_x, True)
    rf_new_X, rf_new_y = gen_pseudo_label(rf_base_models, X, y, test_x)
    nn_new_X, nn_new_y = gen_pseudo_label(nn_base_models, X, y, test_x)

    # pseudo_labeling後の再学習
    cat_base_models, cat_base_preds_valid = train_by_catboost(
        cat_new_X, cat_new_y, kfold
    )
    lgb_base_models, lgb_base_preds_valid = train_by_lightgbm(
        lgb_new_X, lgb_new_y, kfold
    )
    rf_base_models, rf_base_preds_valid = train_by_randomforest(
        rf_new_X, rf_new_y, kfold
    )
    nn_base_models, nn_base_preds_valid = train_by_neuralnet(nn_new_X, nn_new_y, kfold)

    # 各アルゴルにおけるテストデータに対する予測値(metaモデルの学習データ)
    cat_base_preds_test = (
        pd.concat(_predict(cat_base_models, test_x), axis=1).sum(axis=1) / n_splits
    )
    lgb_base_preds_test = (
        pd.concat(_predict(lgb_base_models, test_x), axis=1).sum(axis=1) / n_splits
    )
    rf_base_preds_test = (
        pd.concat(_predict(rf_base_models, test_x), axis=1).sum(axis=1) / n_splits
    )
    nn_base_preds_test = (
        pd.concat(_predict(nn_base_models, test_x), axis=1).sum(axis=1) / n_splits
    )
    cat_base_preds_test = pd.Series(np.where(cat_base_preds_test < 0.5, 0, 1))
    lgb_base_preds_test = pd.Series(np.where(lgb_base_preds_test < 0.5, 0, 1))
    rf_base_preds_test = pd.Series(np.where(rf_base_preds_test < 0.5, 0, 1))
    nn_base_preds_test = pd.Series(np.where(nn_base_preds_test < 0.5, 0, 1))

    meta_test = pd.concat(
        [
            cat_base_preds_test,
            lgb_base_preds_test,
            rf_base_preds_test,
            nn_base_preds_test,
        ],
        axis=1,
    )  # metaモデルのテストデータ
    meta_train = pd.concat(
        [
            cat_base_preds_valid,
            lgb_base_preds_valid,
            rf_base_preds_valid,
            nn_base_preds_valid,
        ],
        axis=1,
    )  # metaモデルの学習データ
    meta_train.columns = [f"col{num}" for num in range(len(meta_train.columns))]
    meta_test.columns = [f"col{num}" for num in range(len(meta_test.columns))]
    return meta_train, meta_test, y


def train_meta(train_x, train_y, kfold):
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        model = CatBoostClassifier(
            iterations=1000,
            # iterations=1,
            learning_rate=0.1,
            use_best_model=True,
            eval_metric="Accuracy",
        )
        model.fit(
            tr_x,
            tr_y,
            eval_set=(val_x, val_y),
        )
        y_pred = model.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)
        models.append(model)
        acc_results.append(accuracy)
    return models, acc_results


def _predict(models, test_x):
    y_preds = []
    for i, model in enumerate(models):
        y_pred = model.predict(test_x)
        pred = [0 if i < 0.5 else 1 for i in y_pred]
        y_preds.append(pd.Series(pred))
    return y_preds


def predict(model, test_x, threshold):
    y_pred = model.predict(test_x)
    pred = [0 if i < threshold else 1 for i in y_pred]
    return pd.Series(pred)


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

    # meta_model = LogisticRegression(
    #     penalty="l2",
    #     C=0.01,
    #     max_iter=200,
    #     verbose=10,
    #     n_jobs=4,
    # )  # 最終結合用モデル

    # ベースモデルでの学習及び推論を行う ⇒ メタデータの学習、検証用データを出力
    meta_train_x, meta_test, meta_train_y = run_train_and_stacking(
        X, y, test_x
    )  # スタッキングの実行

    # metaモデルでの再学習
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    meta_models, acc_results = train_meta(
        train_x=meta_train_x,
        train_y=meta_train_y,
        kfold=kfold,
    )

    # 評価
    threshold = 0.5
    y_preds = []
    for i, model in enumerate(meta_models):
        y_pred = predict(model, meta_test, threshold)
        y_preds.append(y_pred)

    # 提出用ファイル成型
    winner_pred = pd.concat(y_preds, axis=1).mode(axis=1).rename(columns={0: "y"})
    submission = pd.concat([ids, winner_pred], axis=1)

    print(submission)
    print("######################################")
    print(f"accuracy avg = {sum(acc_results) / len(acc_results)}")
    print("######################################")
    submission.to_csv(f"{DATA_DIR}/submission44.csv", index=False)


if __name__ == "__main__":
    run_all()