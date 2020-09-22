import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna  # ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from python_file.kaggle.predict_winner.common import Common
from catboost import CatBoost, Pool, CatBoostClassifier

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


def preprocess(df):
    df = df.drop(REMOVAL_COLS, axis=1)
    categorical_columns = [x for x in df.columns if df[x].dtype == "object"]
    ce_ohe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown="impute")
    encorded_data = ce_ohe.fit_transform(df)

    train = (
        encorded_data[encorded_data["usage"] == 0]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )
    test = (
        encorded_data[encorded_data["usage"] == 1]
        .drop(["usage"], axis=1)
        .reset_index(drop=True)
    )

    ids_col = test.loc[:, "id"]
    train_y = train.loc[:, "y"]
    train_x = train.drop(["y"], axis=1)
    test_x = test.drop(["y"], axis=1)

    return train_x, train_y, test_x, ids_col


def train(train_x, train_y, kfold, best_params=None):
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
        # categorical_columns = [x for x in train_x.columns if train_x[x].dtype == "object"]
        model.fit(
            tr_x,
            tr_y,
            # cat_features=categorical_columns,
            eval_set=(val_x, val_y),
            plot=True,
        )

        y_pred = model.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)
        # # 検証結果の描画
        # fig = lgb.plot_metric(evals_result)
        # plt.savefig(f"{DATA_DIR}/learning_curve_{i+1}.png")

        models.append(model)
        acc_results.append(accuracy)

    return models, acc_results


def accuracy(preds, data, threshold=0.5):
    """精度 (Accuracy) を計算する関数"""
    weight = data.get_weight()
    true_label = data.get_label()
    pred_label = (preds > threshold).astype(int)
    acc = np.average(true_label == pred_label, weights=weight)
    return "accuracy", acc, True


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
    for name in raw_data.columns:
        if "rank" in name:
            raw_data[name] = raw_data[name].map(RANK_MAP)

    # 武器と射程距離の関係分類計算
    buki_range_distance_dict = (
        buki_raw_data.loc[:, ["category2", "key"]]
        .set_index("key")
        .to_dict()["category2"]
    )

    # 新規特徴量追加
    data = cm.calc_team_level_avg(raw_data)
    data = cm.calc_team_rank_avg(data)
    data = cm.add_count_range_distance(data, buki_range_distance_dict)
    data = cm.calc_weapons_win_rate_avg(data, win_rate_dict)
    data = cm.calc_weapons_win_rate_avg_per_mode(data, win_rate_mode_df)

    # lobby-mode列が'regular'のデータ
    regular_data = data[data["lobby-mode"] == "regular"]
    regular_data = regular_data.fillna(regular_data.mode().iloc[0])

    # lobby-mode列が'gachi'のデータ
    gachi_data = data[data["lobby-mode"] == "gachi"]
    gachi_data = gachi_data.fillna(gachi_data.mode().iloc[0])

    # train用とtest用のデータの前処理(regular)
    drop_rank_col = [col for col in regular_data.columns if "rank" not in col]
    regular_data = regular_data.loc[:, drop_rank_col]

    (
        train_x_for_regular,
        train_y_for_regular,
        test_x_for_regular,
        ids_for_regular,
    ) = preprocess(regular_data)

    # train用とtest用のデータの前処理(gachi)
    train_x_for_gachi, train_y_for_gachi, test_x_for_gachi, ids_for_gachi = preprocess(
        gachi_data
    )

    # 学習
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    models_for_regular, acc_results_for_regular = train(
        train_x_for_regular, train_y_for_regular, kfold
    )
    models_for_gachi, acc_results_for_gachi = train(
        train_x_for_gachi, train_y_for_gachi, kfold
    )

    add_data_for_regular = np.zeros((len(test_x_for_regular), 2))
    add_data_for_gachi = np.zeros((len(test_x_for_gachi), 2))
    pseudo_threshold = 0.8

    for model_for_regular, model_for_gachi in zip(models_for_regular, models_for_gachi):
        y_pred_for_regular = model_for_regular.predict_proba(test_x_for_regular)
        y_pred_for_gachi = model_for_gachi.predict_proba(test_x_for_gachi)
        add_data_for_regular += y_pred_for_regular
        add_data_for_gachi += y_pred_for_gachi
    add_data_for_regular = pd.DataFrame(add_data_for_regular / len(models_for_regular))
    add_data_for_gachi = pd.DataFrame(add_data_for_gachi / len(models_for_gachi))

    pseudo_for_regular = pd.concat([ids_for_regular, add_data_for_regular], axis=1)
    pseudo_for_gachi = pd.concat([ids_for_gachi, add_data_for_gachi], axis=1)

    pseudo_data_for_regular = pseudo_for_regular[
        (pseudo_for_regular[0] > pseudo_threshold)
        | (pseudo_for_regular[1] > pseudo_threshold)
    ]
    pseudo_label_for_regular = pd.concat(
        [
            pseudo_data_for_regular.loc[:, "id"],
            pseudo_data_for_regular.drop("id", axis=1).idxmax(axis=1),
        ],
        axis=1,
    ).rename(columns={0: "y"})
    pseudo_data_for_gachi = pseudo_for_gachi[
        (pseudo_for_gachi[0] > pseudo_threshold)
        | (pseudo_for_gachi[1] > pseudo_threshold)
    ]
    pseudo_label_for_gachi = pd.concat(
        [
            pseudo_data_for_gachi.loc[:, "id"],
            pseudo_data_for_gachi.drop("id", axis=1).idxmax(axis=1),
        ],
        axis=1,
    ).rename(columns={0: "y"})

    pseudo_data_for_regular = pd.merge(
        pseudo_label_for_regular,
        test_x_for_regular,
        how="left",
        on=["id"],
    )
    pseudo_data_for_gachi = pd.merge(
        pseudo_label_for_gachi,
        test_x_for_gachi,
        how="left",
        on=["id"],
    )

    new_train_x_for_regular = pd.concat(
        [train_x_for_regular, pseudo_data_for_regular.drop("y", axis=1)], axis=0
    ).reset_index(drop=True)
    new_train_y_for_regular = pd.concat(
        [train_y_for_regular, pseudo_data_for_regular.loc[:, "y"]]
    )
    new_train_x_for_gachi = pd.concat(
        [train_x_for_gachi, pseudo_data_for_gachi.drop("y", axis=1)], axis=0
    ).reset_index(drop=True)
    new_train_y_for_gachi = pd.concat(
        [train_y_for_gachi, pseudo_data_for_gachi.loc[:, "y"]]
    )

    # pseudo_labeling後の再学習
    models_for_regular, acc_results_for_regular = train(
        new_train_x_for_regular, new_train_y_for_regular, kfold
    )
    models_for_gachi, acc_results_for_gachi = train(
        new_train_x_for_gachi, new_train_y_for_gachi, kfold
    )

    # 評価
    threshold = 0.5
    y_preds_for_regular = []
    y_preds_for_gachi = []
    for i, (model_for_regular, model_for_gachi) in enumerate(
        zip(models_for_regular, models_for_gachi)
    ):
        y_pred_for_regular = predict(model_for_regular, test_x_for_regular, threshold)
        y_pred_for_gachi = predict(model_for_gachi, test_x_for_gachi, threshold)
        y_preds_for_regular.append(y_pred_for_regular)
        y_preds_for_gachi.append(y_pred_for_gachi)

    # 提出用ファイル成型
    winner_pred_for_regular = (
        pd.concat(y_preds_for_regular, axis=1)
        .mode(axis=1)
        .mode(axis=1)
        .rename(columns={0: "y"})
    )
    winner_pred_for_gachi = (
        pd.concat(y_preds_for_gachi, axis=1)
        .mode(axis=1)
        .mode(axis=1)
        .rename(columns={0: "y"})
    )

    submission_for_regular = pd.concat(
        [ids_for_regular, winner_pred_for_regular], axis=1
    )
    submission_for_gachi = pd.concat([ids_for_gachi, winner_pred_for_gachi], axis=1)

    submission = pd.concat(
        [submission_for_regular, submission_for_gachi], axis=0
    ).sort_values("id")

    print(submission)
    print("######################################")
    print(
        f"regular accuracy avg = {sum(acc_results_for_regular) / len(acc_results_for_regular)}"
    )
    print(
        f"gachi accuracy avg = {sum(acc_results_for_gachi) / len(acc_results_for_gachi)}"
    )
    print("######################################")
    submission.to_csv(f"{DATA_DIR}/submission35.csv", index=False)


if __name__ == "__main__":
    run_all()