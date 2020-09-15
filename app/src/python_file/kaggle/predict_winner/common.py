from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


WEAPONS = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
    "B1-weapon",
    "B2-weapon",
    "B3-weapon",
    "B4-weapon",
]

WEAPONS_A = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
]

WEAPONS_B = [
    "B1-weapon",
    "B2-weapon",
    "B3-weapon",
    "B4-weapon",
]

BUKI_RANGE_CATEGORY = {
    "charger": 1,
    "splatling": 1,
    "brush": 0,
    "slosher": 0,
    "maneuver": 0,
    "reelgun": 0,
    "brella": 0,
    "roller": 0,
    "blaster": 0,
    "shooter": 0,
}


class Common:
    def __init__(self):
        ...

    def make_feature(self, df_train, df_test):
        mlb = MultiLabelBinarizer()
        mlb.fit([set(df_train["A4-weapon"].fillna("none").unique())])
        train_num = len(df_train)
        df = pd.concat([df_train, df_test])

        # cat_cols = ["lobby-mode", "mode", "stage"]  # 武器以外のカテゴリ変数
        # for c in cat_cols:
        #     vv, obj = pd.factorize(df[c])  # vv: 数値エンコーディングされた値、obj: 列の種類(unique)
        #     df[c] = vv

        A1 = ["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]
        B1 = ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]

        t = mlb.transform(df[A1].fillna("none")[A1].values)
        t2 = mlb.transform(df[B1].fillna("none")[B1].values)

        for i in range(t.shape[1]):
            df["A-" + mlb.classes_[i]] = t[:, i]
            df["B-" + mlb.classes_[i]] = t2[:, i]

        s = ["A", "B"]
        p = ["1", "2", "3", "4"]
        for i in s:
            for j in p:
                df[i + j + "-level"] = df[i + j + "-level"] // 10
                df[i + j + "-level"] = df[i + j + "-level"].clip(0, 30)

        return df[:train_num], df[train_num:]

    def trans_weapon(
        self, df, columns=["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]
    ):
        mlb = MultiLabelBinarizer()
        mlb.fit([set(df["A1-weapon"].unique())])
        MultiLabelBinarizer(classes=None, sparse_output=False)

        weapon = df.fillna("none")
        weapon_binarized = mlb.transform(weapon[columns].values)
        return pd.DataFrame(weapon_binarized, columns=mlb.classes_)

    def make_input_output(self, df, with_y=False):
        # a_weapon = trans_weapon(df, ['A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon'])
        a_weapon = self.trans_weapon(df, ["A2-weapon", "A3-weapon", "A4-weapon"])
        b_weapon = self.trans_weapon(
            df, ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]
        )
        a_weapon = a_weapon.add_suffix("_A")
        b_weapon = b_weapon.add_suffix("_B")
        X = pd.concat([a_weapon, b_weapon], axis=1)
        if with_y:
            y = df["y"]
            return X, y
        return X

    # 武器の勝率を計算する。
    def win_rate(self, buki_name, df):
        count = len(df[df[buki_name + "_A"] == 1]) + len(
            df[df[buki_name + "_B"] == 1]
        )  # それぞれのチームで対象ブキが出現する試合数のカウント
        win = len(df[(df[buki_name + "_A"] == 1) & (df.y == 1)]) + len(
            (df[(df[buki_name + "_B"] == 1) & (df.y == 0)])
        )  # それぞれのチームで対象ブキが出現する試合のうち各チームが勝利する試合数のカウント
        rate = win / count
        return rate, count

    # 各チームで使用している武器の勝率の平均値を算出する
    def calc_weapons_win_rate_avg(self, df, win_rate_dict):
        weapons_rate_A = [
            "A1-weapon_win_rate",
            "A2-weapon_win_rate",
            "A3-weapon_win_rate",
            "A4-weapon_win_rate",
        ]
        weapons_rate_B = [
            "B1-weapon_win_rate",
            "B2-weapon_win_rate",
            "B3-weapon_win_rate",
            "B4-weapon_win_rate",
        ]
        for col in df.columns:
            if col in WEAPONS:
                df[f"{col}_win_rate"] = df[col].map(win_rate_dict)  # 各武器の勝率を新特徴量として追加
        df["team_A_win_rate_avg"] = df.loc[:, weapons_rate_A].mean(axis="columns")
        df["team_B_win_rate_avg"] = df.loc[:, weapons_rate_B].mean(axis="columns")
        # return df
        return df.drop([*weapons_rate_A, *weapons_rate_B], axis=1)  # 平均値のみ特徴量として加える。

    def calc_weapons_win_rate_avg_per_mode(self, df, win_rate_mode_df):
        for mode in df["mode"].unique():
            mode_rate_dict = (
                win_rate_mode_df.loc[:, [f"{mode}_buki", f"{mode}_win_rate"]]
                .set_index(f"{mode}_buki")
                .to_dict()
            )[f"{mode}_win_rate"]
            for col in df.columns:
                if col in WEAPONS:
                    df[f"{col}_win_rate_mode"] = df[col].map(
                        mode_rate_dict
                    )  # 各modeにおける武器の勝率を新特徴量として追加
        return df

    # 各チームにおける遠距離武器人数と近接武器人数の特徴量を追加する(各チーム人数も)。
    def add_count_range_distance(self, df, buki_range_distance_dict):
        buki_category_list = list(set(buki_range_distance_dict.values()))
        count_buki_rangeA = pd.DataFrame()
        count_buki_rangeB = pd.DataFrame()

        teamA_counts = 4 - df.loc[:, WEAPONS_A].isnull().sum(axis=1)
        teamB_counts = 4 - df.loc[:, WEAPONS_B].isnull().sum(axis=1)

        weapons_of_teamA = (
            df.loc[:, WEAPONS_A]
            .replace(buki_range_distance_dict)
            .replace(BUKI_RANGE_CATEGORY)
        ).sum(axis=1)

        count_buki_rangeA["count_long_distance_A"] = weapons_of_teamA
        count_buki_rangeA["count_short_distance_A"] = teamA_counts - weapons_of_teamA
        count_buki_rangeA["count_team_A"] = weapons_of_teamA

        weapons_of_teamB = (
            df.loc[:, WEAPONS_B]
            .replace(buki_range_distance_dict)
            .replace(BUKI_RANGE_CATEGORY)
        ).sum(axis=1)
        count_buki_rangeB["count_long_distance_B"] = weapons_of_teamB
        count_buki_rangeB["count_short_distance_B"] = teamB_counts - weapons_of_teamB
        count_buki_rangeB["count_team_B"] = weapons_of_teamB
        return pd.concat([df, count_buki_rangeA, count_buki_rangeB], axis=1)

    # 各チームのレベル平均を算出する。
    def calc_team_level_avg(self, df):
        A_level_list = ["A1-level", "A2-level", "A3-level", "A4-level"]
        B_level_list = ["B1-level", "B2-level", "B3-level", "B4-level"]
        df["teamA_level_avg"] = df.loc[:, A_level_list].mean(axis=1)
        df["teamB_level_avg"] = df.loc[:, B_level_list].mean(axis=1)
        return df

    # 各チームのランク平均を算出する。
    def calc_team_rank_avg(self, df):
        A_rank_list = ["A1-rank", "A2-rank", "A3-rank", "A4-rank"]
        B_rank_list = ["B1-rank", "B2-rank", "B3-rank", "B4-rank"]
        df["teamA_rank_avg"] = df.loc[:, A_rank_list].mean(axis=1)
        df["teamB_rank_avg"] = df.loc[:, B_rank_list].mean(axis=1)
        return df