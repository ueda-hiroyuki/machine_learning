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

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
BUKI_PATH = f"{DATA_DIR}/weapon.csv"

REMOVAL_COLS = ["lobby", "game-ver"]

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
    categorical_columns = [x for x in df.columns if df[x].dtype == "object"]

    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown="impute")
    encorded_data = ce_oe.fit_transform(df)

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

    # cf.check_corr(train, "predict_winner")

    train_y = train.loc[:, "y"]
    train_x = train.drop(["y"], axis=1)
    test_x = test.drop(["y"], axis=1)

    return train_x, train_y, test_x


def train(train_x, train_y, kfold, best_params=None):
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

    data = cm.calc_team_level_avg(raw_data)
    data = cm.calc_team_rank_avg(data)
    data = cm.add_count_range_distance(data, buki_range_distance_dict)

    # lobby-mode列が'regular'のデータ
    regular_data = data[data["lobby-mode"] == "regular"].drop(REMOVAL_COLS, axis=1)
    regular_data = regular_data.fillna(regular_data.mode().iloc[0])

    # lobby-mode列が'gachi'のデータ
    gachi_data = data[data["lobby-mode"] == "gachi"].drop(REMOVAL_COLS, axis=1)
    gachi_data = gachi_data.fillna(gachi_data.mode().iloc[0])

    # train用とtest用のデータの前処理(regular)
    drop_rank_col = [col for col in regular_data.columns if "rank" not in col]
    regular_data = regular_data.loc[:, drop_rank_col]

    train_x_for_regular, train_y_for_regular, test_x_for_regular = preprocess(
        regular_data
    )
    ids_for_regular = test_x_for_regular.loc[:, "id"]

    # train用とtest用のデータの前処理(gachi)
    train_x_for_gachi, train_y_for_gachi, test_x_for_gachi = preprocess(gachi_data)
    ids_for_gachi = test_x_for_gachi.loc[:, "id"]

    # 学習
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    models_for_regular, acc_results_for_regular = train(
        train_x_for_regular, train_y_for_regular, kfold
    )
    models_for_gachi, acc_results_for_gachi = train(
        train_x_for_gachi, train_y_for_gachi, kfold
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
    submission.to_csv(f"{DATA_DIR}/submission23.csv", index=False)


if __name__ == "__main__":
    run_all()