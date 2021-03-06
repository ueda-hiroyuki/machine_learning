import itertools
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna  # ハイパーパラメータチューニング自動化ライブラリ
from tqdm import tqdm
from functools import partial
from catboost import CatBoost, Pool, CatBoostClassifier
from optuna.integration import lightgbm_tuner
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV, RFE
from python_file.kaggle.predict_winner.common import Common
from python_file.kaggle.predict_winner.params_map import (
    WEAPON_MAP,
    RANK_MAP,
    STAGE_AREA_MAP,
    BUKI_MAP,
)

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
BUKI_PATH = f"{DATA_DIR}/weapon.csv"
REMOVAL_COLS = ["lobby", "game-ver", "period"]


cm = Common()


def train(train_x, train_y, kfold, best_params=None):
    categorical_columns = [x for x in train_x.columns if train_x[x].dtype == "object"]
    models = []
    acc_results = []
    for i, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        model = CatBoostClassifier(
            iterations=1000,
            use_best_model=True,
            learning_rate=0.1,
            eval_metric="Accuracy",
        )
        model.fit(tr_x, tr_y, eval_set=(val_x, val_y))

        # model = CatBoostClassifier(
        #     iterations=best_params["iterations"],
        #     use_best_model=True,
        #     depth=best_params["depth"],
        #     learning_rate=best_params["learning_rate"],
        #     l2_leaf_reg=best_params["l2_leaf_reg"],
        #     random_strength=best_params["random_strength"],
        #     eval_metric="Accuracy",
        # )

        y_pred = model.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)

        models.append(model)
        acc_results.append(accuracy)

    return models, acc_results


def expand_dataset(N: int, df: pd.DataFrame, random_state=42):
    """データを拡張する"""

    if N == 0:
        return df

    # 実際には列名のリストが格納されている
    a1 = ["A1-weapon", "A1-rank", "A1-level"]
    a2 = ["A2-weapon", "A2-rank", "A2-level"]
    a3 = ["A3-weapon", "A3-rank", "A3-level"]
    a4 = ["A4-weapon", "A4-rank", "A4-level"]
    b1 = ["B1-weapon", "B1-rank", "B1-level"]
    b2 = ["B2-weapon", "B2-rank", "B2-level"]
    b3 = ["B3-weapon", "B3-rank", "B3-level"]
    b4 = ["B4-weapon", "B4-rank", "B4-level"]

    train = df.copy(deep=True)
    train_temp = df.copy(deep=True)
    all_train = df.copy(deep=True)

    a_team_list = [a1, a2, a3, a4]
    b_team_list = [b1, b2, b3, b4]

    a_team_combination = list(itertools.permutations(a_team_list))
    b_team_combination = list(itertools.permutations(b_team_list))

    a_and_b_combination = list(
        itertools.product(a_team_combination, b_team_combination)
    )
    if N > len(a_and_b_combination):
        pattern_list = a_and_b_combination
    else:
        random.seed(random_state)
        pattern_list = random.sample(a_and_b_combination, N)
    for pattern in pattern_list:
        train[pattern[0][0]] = train_temp[a1]
        train[pattern[0][1]] = train_temp[a2]
        train[pattern[0][2]] = train_temp[a3]
        train[pattern[0][3]] = train_temp[a4]
        train[pattern[1][0]] = train_temp[b1]
        train[pattern[1][1]] = train_temp[b2]
        train[pattern[1][2]] = train_temp[b3]
        train[pattern[1][3]] = train_temp[b4]

        all_train = pd.concat([all_train, train], axis=0)
    return all_train


def get_important_features(train_x, train_y):
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, random_state=1)
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        use_best_model=True,
        eval_metric="Accuracy",
    )
    model.fit(
        tr_x,
        tr_y,
        eval_set=(val_x, val_y),
        plot=True,
    )
    importance = pd.DataFrame(
        model.get_feature_importance(), index=train_x.columns, columns=["importance"]
    ).sort_values("importance", ascending=False)
    print(importance)
    return importance


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

    train_raw_data["stage_area"] = train_raw_data["stage"].replace(STAGE_AREA_MAP)
    test_raw_data["stage_area"] = test_raw_data["stage"].replace(STAGE_AREA_MAP)
    train_buki_ability = cm.add_buki_ability(train_raw_data)
    test_buki_ability = cm.add_buki_ability(test_raw_data)
    buki_ability = pd.concat(
        [train_buki_ability, test_buki_ability], axis=0
    ).reset_index(drop=True)

    train_raw_data = expand_dataset(20, train_raw_data).reset_index(drop=True)

    test_raw_data["y"] = 0
    train_raw_data["usage"] = 0  # for train
    test_raw_data["usage"] = 1  # for test

    # X = cm.make_input_output(train_raw_data, with_y=False)
    # train_data = pd.concat([train_raw_data, X], axis=1)

    train_data = train_raw_data

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
    data = pd.concat([data, buki_ability], axis=1)

    categorical_columns = [x for x in data.columns if data[x].dtype == "object"]

    ce_ohe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown="impute")
    encoded_data = ce_ohe.fit_transform(data)

    encoded_data = cf.corr_column(encoded_data, 0.6)  # 相関の強い列を削除

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

    # 学習
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    models, acc_results = train(train_x, train_y, kfold)

    # pseudo_labeling
    add_data = np.zeros((len(test_x), 2))
    for i, model in enumerate(models):
        y_pred = model.predict_proba(test_x)
        add_data += y_pred
    add_data = pd.DataFrame(add_data / len(models))
    pseudo_label = add_data[(add_data[0] > 0.7) | (add_data[1] > 0.7)].idxmax(
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
    models, acc_results = train(new_train_x, new_train_y, kfold)

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
    print(f"pseudo before:{len(train_x)}, after:{len(new_train_x)}")
    print(f"accuracy avg = {sum(acc_results) / len(acc_results)}")
    print("######################################")
    submission.to_csv(f"{DATA_DIR}/submission60.csv", index=False)


if __name__ == "__main__":
    run_all()
