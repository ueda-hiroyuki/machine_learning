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

    def win_rate(self, buki_name, df):
        count = len(df[df[buki_name + "_A"] == 1]) + len(
            df[df[buki_name + "_B"] == 1]
        )  # それぞれのチームで対象ブキが出現する試合数のカウント
        win = len(df[(df[buki_name + "_A"] == 1) & (df.y == 1)]) + len(
            (df[(df[buki_name + "_B"] == 1) & (df.y == 0)])
        )  # それぞれのチームで対象ブキが出現する試合のうち各チームが勝利する試合数のカウント
        rate = win / count
        return rate, count

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
        return df

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
