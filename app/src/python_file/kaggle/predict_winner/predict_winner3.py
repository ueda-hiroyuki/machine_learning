import joblib
import trueskill as ts
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna  # ハイパーパラメータチューニング自動化ライブラリ
from tqdm import tqdm
from optuna.integration import lightgbm_tuner
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score


DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
REMOVAL_COLS = ["lobby", "game-ver", "period"]

rank_map = {
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

weapon_cols = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
    "B1-weapon",
    "B2-weapon",
    "B3-weapon",
    "B4-weapon",
]


def calc_weapon_rate(df, mode):

    """
    各modeにおける武器レーティングの算出
    """
    env = ts.TrueSkill()
    weapon_dict = {}

    if mode != "all":
        _df = df[df["mode"] == mode]
    else:
        _df = df.copy()

    # 両チーム4人そろっているものを対象
    _df = _df[
        ~(
            _df[
                [
                    "A1-weapon",
                    "A2-weapon",
                    "A3-weapon",
                    "A4-weapon",
                    "B1-weapon",
                    "B2-weapon",
                    "B3-weapon",
                    "B4-weapon",
                ]
            ]
            .isnull()
            .sum(axis=1)
            > 0
        )
    ]

    # 全員のランクが同一のバトルを対象
    if mode not in ["nawabari", "all"]:
        _df = _df[
            (_df["A1-rank"] == _df["A2-rank"])
            & (_df["A1-rank"] == _df["A3-rank"])
            & (_df["A1-rank"] == _df["A4-rank"])
            & (_df["A1-rank"] == _df["B1-rank"])
            & (_df["A1-rank"] == _df["B2-rank"])
            & (_df["A1-rank"] == _df["B3-rank"])
            & (_df["A1-rank"] == _df["B4-rank"])
        ]

    for idx, row in tqdm(_df.sort_values("period").iterrows(), total=len(_df)):
        team_a = {}
        team_b = {}

        for weapon_column in ["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]:
            weapon = row[weapon_column]
            team_a[weapon] = weapon_dict.get(weapon, env.create_rating())

        for weapon_column in ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]:
            weapon = row[weapon_column]
            team_b[weapon] = weapon_dict.get(weapon, env.create_rating())

        team_a, team_b, = env.rate(
            (
                team_a,
                team_b,
            ),
            ranks=(
                abs(row["y"] - 1),
                row["y"],
            ),
        )

        weapon_dict.update(team_a)
        weapon_dict.update(team_b)

    rate_dict = {k: float(v) for k, v in weapon_dict.items()}

    return rate_dict


def preprocess(train_path, test_path):
    train_raw_data = pd.read_csv(train_path)
    test_raw_data = pd.read_csv(test_path)

    test_raw_data["y"] = 0
    train_raw_data["usage"] = 0  # for train
    test_raw_data["usage"] = 1  # for test

    raw_data = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)

    # filterd_raw_data = raw_data.drop(
    #     [x for x in raw_data.columns if "rank" in x],
    #     axis=1,  # 'rank'を含んだ列は何nanが多いため、とりあえず除去
    # )
    filterd_raw_data = raw_data
    filterd_raw_data = filterd_raw_data.fillna(filterd_raw_data.mode().iloc[0]).drop(
        REMOVAL_COLS, axis=1
    )
    dfs = []
    for mode in ["nawabari", "area", "asari", "hoko", "yagura"]:
        weapons_rating = joblib.load(f"{DATA_DIR}/weapons_rating_{mode}.pkl")
        extracted = filterd_raw_data[filterd_raw_data["mode"] == mode]
        rate_map = {}
        for _, row in weapons_rating.iterrows():
            rate_map[row.weapon] = row.rating
        for col in weapon_cols:
            extracted[col] = extracted[col].map(rate_map)
        dfs.append(extracted)

    mapped_data = pd.concat(dfs, axis=0).reset_index(drop=True)

    for name in mapped_data.columns:
        if "rank" in name:
            mapped_data[name] = mapped_data[name].map(rank_map)

    categorical_columns = [
        x for x in mapped_data.columns if mapped_data[x].dtype == "object"
    ]

    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown="impute")
    encorded_data = ce_oe.fit_transform(mapped_data)

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

    train_y = train.loc[:, "y"].reset_index(drop=True)
    train_x = train.drop(["y"], axis=1).reset_index(drop=True)
    test_x = test.drop(["y"], axis=1).sort_values("id").reset_index(drop=True)

    return train_x, train_y, test_x


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
        early_stopping_rounds=20,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history,
    )
    return best_params


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
            num_boost_round=1000,
            early_stopping_rounds=20,
            verbose_eval=1,
            evals_result=evals_result,
            # feval=accuracy,
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

    return models


def accuracy(preds, data, threshold=0.5):
    """精度 (Accuracy) を計算する関数"""
    weight = data.get_weight()
    true_label = data.get_label()
    pred_label = (preds > threshold).astype(int)
    acc = np.average(true_label == pred_label, weights=weight)
    return "accuracy", acc, True


def predict(model, test_x, threshold):
    y_pred = model.predict(test_x, num_iteration=model.best_iteration)
    return pd.Series(y_pred)


def run_all():
    # train用とtest用のデータの前処理
    train_x, train_y, test_x = preprocess(TRAIN_PATH, TEST_PATH)

    ids = test_x.loc[:, "id"]

    # # 学習用のハイパラをチューニング
    # best_params = get_best_params(train_x, train_y)
    # print(best_params)

    # 学習
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    models = train(train_x, train_y, kfold)

    # 評価
    threshold = 0.5
    y_preds = pd.Series(np.zeros(len(test_x)), name="y")
    for i, model in enumerate(models):
        y_pred = predict(model, test_x, threshold)
        y_preds += y_pred

    # 提出用ファイル成型
    winner_pred = np.where(y_preds / n_splits > threshold, 1, 0)
    submission = pd.concat([ids, pd.Series(winner_pred, name="y")], axis=1)
    print(submission)

    submission.to_csv(f"{DATA_DIR}/submission8.csv", index=False)


if __name__ == "__main__":
    run_all()