import itertools
import random
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
    # print(a_and_b_combination)

    if N > len(a_and_b_combination):
        pattern_list = a_and_b_combination
    else:
        random.seed(random_state)
        pattern_list = random.sample(a_and_b_combination, N)
    print(pattern_list[0], len(pattern_list))
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


def solution_7th():
    # 集計系特徴量の追加(チーム内の合計、標準偏差、尖度、チーム毎の差、比率：rank,levelにおいて)
    # null importance による特徴量選択
    ...


def solution_8th():
    # 外部から取ってきた武器の数値データについて、チーム内での基本統計量を算出
    # チームを入れ替えることによるデータセットの水増し
    # 武器の、単純な使用回数、モードごとでの使用回数をカウントエンコーディング
    # 武器（stat.inkのcategory1, category2, mainweapon, subweapon, spacial）のワンホットエンコーディング
    # チーム編成（stat.inkのcategory1, category2, mainweapon, subweapon, spacial）のカテゴリ変数化
    # mode, stageとその他のカテゴリ変数の直積
    ...

def solution_9th():
    # ⓵カテゴリ変数をターゲットエンコーディングしたもの
    # ⓶⓵の特徴量をfeature toolsで水増ししたもの(足し算、引き算、掛け算)
    # ⓷武器ラベルはワンホットエンコーディング(MultiLabelBinarizer) ⇒ それ以外の変数はfactrize()を使用
    # ④6つのモデルのアンサンブル(加重平均) ⇒ GaussianNBとBernoulliNBのstacking、lightgbm、logisticregression、XGBoost(前処理との組み合わせ)



def solution_6th():
    # 新規特徴量などは無し。
    # 10分割CV、lightgbmシングルモデル
    # A1～A4とB1～B4を入れ替えて、ラベルを反転させる(学習データの水増し)
    all_train = expand_dataset(10, raw_data)
    print(all_train)


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
    rensen_count = (
        melted.groupby(["period", "mode", "level", "weapon_name"])
        .size()
        .rename("rensen_count")
    )
    weapon_count = (
        melted.groupby(["period", "mode", "level"])["weapon_name"]
        .nunique()
        .rename("weapon_count")
    )
    # print("#######################")
    # print(melted)
    # print("#######################")
    # print(rensen_count)

    # print("#######################")
    # print(weapon_count)

    merged = melted.merge(
        rensen_count, on=["period", "mode", "level", "weapon_name"], how="left"
    )
    merged = merged.merge(weapon_count, on=["period", "mode", "level"], how="left")
    print("#######################")
    print(merged)
    pivoted = merged.pivot(
        columns=["period", "mode"], values=["level", "rensen_count", "weapon_count"]
    )
    print("#######################")
    print(pivoted)


if __name__ == "__main__":
    solution_6th()