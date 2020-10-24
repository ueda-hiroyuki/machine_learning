import itertools
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import (
    LabelEncoder,
    LabelBinarizer,
    OrdinalEncoder,
    MultiLabelBinarizer,
)


weapons = [
    "A1-weapon",
    "A2-weapon",
    "A3-weapon",
    "A4-weapon",
]

DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"

train_raw_data = pd.read_csv(TRAIN_PATH)
test_raw_data = pd.read_csv(TEST_PATH)

test_raw_data["y"] = 0
train_raw_data["usage"] = 0  # for train
test_raw_data["usage"] = 1  # for test
raw_data = pd.concat([train_raw_data, test_raw_data], axis=0).reset_index(drop=True)


def encoding_method():
    df = train_raw_data.loc[:, weapons].head(3)

    mlb = MultiLabelBinarizer()
    result = mlb.fit_transform(df.values)
    df_trans = pd.DataFrame(result, columns=mlb.classes_)
    print(df)
    print(df_trans)


def null_importance():
    def display_distributions(actual_imp_df, null_imp_df, feature, num):
        # ある特徴量に対する重要度を取得
        actual_imp = actual_imp_df.query(f"feature == '{feature}'")["importance"].mean()
        null_imp = null_imp_df.query(f"feature == '{feature}'")["importance"]

        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        a = ax.hist(null_imp, label="Null importances")
        ax.vlines(
            x=actual_imp,
            ymin=0,
            ymax=np.max(a[0]),
            color="r",
            linewidth=10,
            label="Real Target",
        )
        ax.legend(loc="upper right")
        ax.set_title(f"Importance of {feature.upper()}", fontweight="bold")
        plt.xlabel(f"Null Importance Distribution for {feature.upper()}")
        plt.ylabel("Importance")
        plt.savefig(f"{DATA_DIR}/{num}th_importance.png")

    def train(X, y):
        clf = lgb.LGBMClassifier()
        clf.fit(X, y)
        # 特徴量の重要度を含むデータフレームを作成
        imp_df = pd.DataFrame()
        imp_df["feature"] = X.columns
        imp_df["importance"] = clf.feature_importances_
        return imp_df.sort_values("importance", ascending=False)

    data = raw_data.fillna(raw_data.mode().iloc[0])
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

    train_y = train_data.loc[:, "y"]
    train_x = train_data.drop(["y"], axis=1)
    test_x = test_data.drop(["y"], axis=1)

    # 特徴量の重要度を含むデータフレームを作成(本来のimportance)
    actual_imp_df = train(train_x, train_y)

    # null importanceの作成
    trials = 100
    null_imp_df = pd.DataFrame()
    for i in range(trials):
        # 目的変数をシャッフルする
        y_permuted = np.random.permutation(train_y)
        imp_df = train(train_x, y_permuted)
        imp_df["run"] = i + 1
        null_imp_df = pd.concat([null_imp_df, imp_df])

    # 閾値を設定
    THRESHOLD = 90
    # 閾値を超える特徴量を取得
    imp_features = []
    for num, feature in enumerate(actual_imp_df["feature"]):
        display_distributions(
            actual_imp_df, null_imp_df, feature, num
        )  # null importance ヒストグラム作成
        actual_value = actual_imp_df.query(f"feature=='{feature}'")["importance"].values
        null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
        percentage = (null_value < actual_value).sum() / null_value.size * 100
        if percentage >= THRESHOLD:
            imp_features.append(feature)
    print(imp_features)


if __name__ == "__main__":
    null_importance()