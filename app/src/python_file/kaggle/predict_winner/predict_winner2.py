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


DATA_DIR = "src/sample_data/Kaggle/predict_winner"
TRAIN_PATH = f"{DATA_DIR}/train_data.csv"
TEST_PATH = f"{DATA_DIR}/test_data.csv"
REMOVAL_COLS = ["lobby", "game-ver"]


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

    categorical_columns = [
        x for x in filterd_raw_data.columns if filterd_raw_data[x].dtype == "object"
    ]

    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown="impute")
    encorded_data = ce_oe.fit_transform(filterd_raw_data)

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
    pred = [0 if i < threshold else 1 for i in y_pred]
    return pd.Series(pred)


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
    y_preds = []
    for i, model in enumerate(models):
        y_pred = predict(model, test_x, threshold)
        y_preds.append(y_pred)

    # 提出用ファイル成型
    winner_pred = pd.concat(y_preds, axis=1).mode(axis=1).rename(columns={0: "y"})
    submission = pd.concat([ids, winner_pred], axis=1)

    print(submission)
    submission.to_csv(f"{DATA_DIR}/submission5.csv", index=False)


if __name__ == "__main__":
    run_all()