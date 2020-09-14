import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna  # ハイパーパラメータチューニング自動化ライブラリ
from tqdm import tqdm
from functools import partial
from optuna.integration import lightgbm_tuner
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV, RFE
from python_file.kaggle.predict_winner.common import Common

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
BUKI_PATH = f"{DATA_DIR}/weapon.csv"
REMOVAL_COLS = ["lobby", "game-ver", "period", "id"]

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


def get_best_params(train_x, train_y):
    tr_x, val_x, tr_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=1
    )
    tr_set = lgb.Dataset(tr_x, tr_y)
    val_set = lgb.Dataset(val_x, val_y, reference=tr_set)
    best_params = {}
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        tr_set,
        valid_sets=val_set,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history,
    )
    return best_params


def get_important_features(train_x, train_y):
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, random_state=1)
    gbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_jobs=4,
        n_estimators=1000,
    )
    result = gbm.fit(tr_x, tr_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100)
    importance = pd.DataFrame(
        gbm.feature_importances_, index=train_x.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    return importance


def objective(X, y, trial):
    """最適化する目的関数"""
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
    gbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.1,
    )
    # RFE で取り出す特徴量の数を最適化する
    n_features_to_select = (
        trial.suggest_int("n_features_to_select", 100, len(list(tr_x.columns))),
    )
    rfe = RFE(estimator=gbm, n_features_to_select=n_features_to_select)
    rfe.fit(tr_x, tr_y)
    selected_cols = list(tr_x.columns[rfe.support_])

    tr_x_selected = tr_x.loc[:, selected_cols]
    val_x_selected = val_x.loc[:, selected_cols]
    gbm.fit(
        tr_x_selected,
        tr_y,
        eval_set=[(val_x_selected, val_y)],
        early_stopping_rounds=100,
    )
    y_pred = gbm.predict(val_x_selected)
    accuracy = accuracy_score(val_y, y_pred)
    return accuracy


def train(train_x, train_y, kfold, best_params=None):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": {"binary_logloss"},
        "num_leaves": 50,
        "min_data_in_leaf": 100,
        "learning_rate": 0.1,
        "feature_fraction": 0.7,
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
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=1,
            evals_result=evals_result,
            feval=accuracy,
        )

        importance = pd.DataFrame(
            model.feature_importance(), index=train_x.columns, columns=["importance"]
        ).sort_values("importance", ascending=[False])

        # print(f"######################importance#####################")
        # print(importance.head(50))

        # 検証結果の描画
        fig = lgb.plot_metric(evals_result)
        plt.savefig(f"{DATA_DIR}/learning_curve_{i+1}.png")

        models.append(model)
        acc_results.append(max(evals_result["eval"]["accuracy"]))

    return models, acc_results


def accuracy(preds, data, threshold=0.5):
    """精度 (Accuracy) を計算する関数"""
    weight = data.get_weight()
    true_label = data.get_label()
    pred_label = (preds > threshold).astype(int)
    acc = np.average(true_label == pred_label, weights=weight)
    return "accuracy", acc, True


def predict(model, test_x, threshold):
    y_pred = model.predict(test_x, num_iteration=model.best_iteration)
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
        # print(buki, win_rate(buki))
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
    raw_data = raw_data.fillna(raw_data.mode().iloc[0])

    data = raw_data
    # data = cm.calc_team_level_avg(raw_data)
    # data = cm.add_count_range_distance(data, buki_range_distance_dict)
    # data = cm.calc_weapons_win_rate_avg(data, win_rate_dict)
    # data = cm.calc_weapons_win_rate_avg_per_mode(data, win_rate_mode_df)

    categorical_columns = [x for x in data.columns if data[x].dtype == "object"]

    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown="impute")
    encorded_data = ce_oe.fit_transform(data)

    train_data = (
        encorded_data[encorded_data["usage"] == 0]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )
    test_data = (
        encorded_data[encorded_data["usage"] == 1]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )

    ids = test_data.loc[:, "id"]

    train_y = train_data.loc[:, "y"]
    train_x = train_data.drop(["y"], axis=1).drop(REMOVAL_COLS, axis=1)
    test_x = test_data.drop(["y"], axis=1).drop(REMOVAL_COLS, axis=1)

    # importance結果を算出
    importance = get_important_features(train_x, train_y)
    selected_feature = list(importance.head(round(len(importance) * 0.1)).index)

    selected_train_x = train_x.loc[:, selected_feature]
    selected_test_x = test_x.loc[:, selected_feature]

    # # 学習用のハイパラをチューニング
    # best_params = get_best_params(train_x, train_y)

    # 学習
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    models, acc_results = train(selected_train_x, train_y, kfold)

    # 評価
    threshold = 0.5
    y_preds = []
    for i, model in enumerate(models):
        y_pred = predict(model, selected_test_x, threshold)
        y_preds.append(y_pred)

    # 提出用ファイル成型
    winner_pred = pd.concat(y_preds, axis=1).mode(axis=1).rename(columns={0: "y"})
    submission = pd.concat([ids, winner_pred], axis=1)

    print(submission)
    print("######################################")
    print(f"accuracy avg = {sum(acc_results) / len(acc_results)}")
    print("######################################")
    submission.to_csv(f"{DATA_DIR}/submission21.csv", index=False)


if __name__ == "__main__":
    run_all()
