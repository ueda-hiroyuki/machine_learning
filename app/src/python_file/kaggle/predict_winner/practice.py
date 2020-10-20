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


teams_a = ["A1", "A2", "A3", "A4"]
teams_b = ["B1", "B2", "B3", "B4"]
persons = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
weapons = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
    "B1-weapon",
    "B2-weapon",
    "B3-weapon",
    "B4-weapon",
]
levels = [
    "A1-level",
    "A2-level",
    "A3-level",
    "A4-level",
    "B1-level",
    "B2-level",
    "B3-level",
    "B4-level",
]
ranks = [
    "A1-rank",
    "A2-rank",
    "A3-rank",
    "A4-rank",
    "B1-rank",
    "B2-rank",
    "B3-rank",
    "B4-rank",
]
player_columns = ["weapon", "rank", "level"]
common_columns = ["id", "period", "mode", "stage"]

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"

train_raw_data = pd.read_csv(TRAIN_PATH)
test_raw_data = pd.read_csv(TEST_PATH)

test_raw_data["y"] = 0
train_raw_data["usage"] = 0  # for train
test_raw_data["usage"] = 1  # for test
raw_data = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)


def solution_2th():
    # 仮説⓵：一定期間、同じ武器を使い続けているプレイヤーは調子が良い(勝ち続けている)のでは？
    # ⇒ mode, level, weapon_nameが同じであれば同一プレイヤーとみなす。
    # 仮説⓶：一定期間負け続けているときは、プレイヤーが使用する武器の個数(種類)が多いのでは？

    dataset1 = raw_data.drop(
        [
            "id",
            "game-ver",
            "lobby",
            "lobby-mode",
            "stage",
            "y",
            "usage",
            *ranks,
            *levels,
        ],
        axis=1,
    )
    dataset2 = raw_data.drop(
        [
            "id",
            "game-ver",
            "lobby",
            "lobby-mode",
            "stage",
            "y",
            "usage",
            *ranks,
            *weapons,
        ],
        axis=1,
    )
    melted1 = pd.melt(
        dataset1,
        id_vars=["period", "mode"],
        var_name="weapon_label",
        value_name="weapon_name",
    ).drop("weapon_label", axis=1)
    melted2 = pd.melt(
        dataset2, id_vars=["period", "mode"], var_name="level_label", value_name="level"
    ).drop(["level_label", "period", "mode"], axis=1)
    melted = pd.concat([melted1, melted2], axis=1)
    print(melted)
    rensen_count = (
        melted.groupby(["period", "mode", "level", "weapon_name"])
        .size()
        .rename("rensen_count")
    )
    merged = melted.merge(
        rensen_count, on=["period", "mode", "level", "weapon_name"], how="left"
    )
    print(merged)


if __name__ == "__main__":
    solution_2th()