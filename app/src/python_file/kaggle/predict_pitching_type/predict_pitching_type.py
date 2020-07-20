import gc
import logging
import collections
import typing as t
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import category_encoders as ce # カテゴリ変数encording用ライブラリ
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.impute import SimpleImputer 
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score


"""
train_pitch(51 columns)
test_pitch(49 columns)
2つの差は「球種」と「投球位置区域」である
⇒今回は「球種」を分類する(0~7の8種類)。
train_player(25 columns)
test_player(25 columns)
・pitchとplayerで共通しているcolumn：「年度」,
・submission.csvに記載する情報は「test_pitchのデータ連番」「各球種(8種)の投球確率」
"""

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/predict_pitching_type"
EXTERNAL_DATA_DIR = f"{DATA_DIR}/external_data"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
EXTERNAL_1_PATH = f"{EXTERNAL_DATA_DIR}/2017登録投手の昨年球種配分.csv"
EXTERNAL_2_PATH = f"{EXTERNAL_DATA_DIR}/2018登録投手の昨年球種配分.csv"
EXTERNAL_3_PATH = f"{EXTERNAL_DATA_DIR}/2019登録投手の昨年球種配分.csv"

PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数", "試合ID"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    y_true = data.get_label()
    y_preds = np.reshape(preds, [len(y_true), 8], order='F')
    y_pred = np.argmax(y_preds, axis=1)
    metric = np.mean(y_true == y_pred)
    return 'accuracy', metric, True

def get_best_params(train_x: t.Any, train_y: t.Any, num_class: int) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y)
    best_params = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'num_class': num_class,
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=10000,
        early_stopping_rounds=20,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history
    )
    return best_params

def get_model(tr_dataset: t.Any, val_dataset: t.Any, num_class: int, best_params: t.Dict[str, t.Any]) -> t.Any:
    evals_result = {}
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate' : 0.1,
        'num_class': num_class,
        **best_params
    }
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        num_boost_round=1000,
        # learning_rates=lambda iter: 0.1 * (0.99 ** iter),
        callbacks=[lgb.reset_parameter(learning_rate=[0.1] * 600 + [0.01] * 400)],
        early_stopping_rounds=100,
        verbose_eval=10,
        evals_result=evals_result,
    )
    return model, evals_result

def objective(X, y, trial):
    """最適化する目的関数"""
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        n_jobs = 4,
        n_estimators=10000,
    )
    n_components = trial.suggest_int('n_components', 1, len(list(X.columns))),
    pca = PCA(n_components=n_components[0]).fit(X)
    x_pca = pca.transform(X)
    tr_x, val_x, tr_y, val_y = train_test_split(x_pca, y, random_state=1)
    gbm.fit(
        tr_x, 
        tr_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=20,
        verbose=50
    )
    y_pred = gbm.predict(val_x)
    accuracy = accuracy_score(val_y, y_pred)
    return accuracy

def get_important_features(train_x: t.Any, train_y: t.Any, best_feature_count: int):
    pca = PCA(n_components=best_feature_count).fit(train_x)
    x_pca = pca.transform(train_x)
    return x_pca, train_y


def main():
    """
    << 処理の流れ >>
    データ読み込み ⇒ 投球データと選手データの結合(train,testも結合) ⇒ nanの置換 ⇒ カテゴリ変数の変換 ⇒
    RFEによる特徴量選択(個数の最適化) ⇒ ハイパーパラメータの最適化 ⇒ 交差検証
    """

    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)

    pitching_type_2016 = pd.read_csv(EXTERNAL_1_PATH)
    pitching_type_2017 = pd.read_csv(EXTERNAL_2_PATH)
    pitching_type_2018 = pd.read_csv(EXTERNAL_3_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test"
    test_pitch["球種"] = 0
    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)
    pitching_type_ratio = pd.concat([pitching_type_2016, pitching_type_2017, pitching_type_2018], axis=0).reset_index(drop=True)

    merged = pd.merge(
        pitch_data, 
        player_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)
    merged = merged.rename(columns={"選手名": "投手名", "チーム名": "投手チーム名"})

    # データセットと前年度投球球種割合をmergeする
    merged = pd.merge(
        merged,
        pitching_type_ratio,
        how="left",
        left_on=['年度','投手ID', "投手名"],
        right_on=['年度','選手ID', "選手名"]
    ).drop(['選手ID', "選手名"], axis=1)

    use = merged.loc[:, "use"]
    merged = merged.drop(["use", "位置", "年度"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged.columns if merged[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged) 
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    print(train_y.value_counts())

    sm = SMOTE(
        ratio={
            0:sum(train_y==0), 
            1:sum(train_y==1)*3,
            2:sum(train_y==2),
            3:sum(train_y==3)*2,
            4:sum(train_y==4)*2,
            5:sum(train_y==5)*4,
            6:sum(train_y==6)*20,
            7:sum(train_y==7)*4
        }
    )
    train_x_resampled, train_y_resampled = sm.fit_sample(train_x, train_y)
    train_x_resampled = pd.DataFrame(train_x_resampled, columns=train_x.columns)
    train_y_resampled = pd.Series(train_y_resampled, name="球種")

    # f = partial(objective, train_x, train_y) # 目的関数に引数を固定しておく
    # study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する

    # study.optimize(f, n_trials=10) # 試行回数を決定する
    # print('params:', study.best_params)# 発見したパラメータを出力する
    # best_feature_count = study.best_params['n_components']
    best_feature_count = 47
    # x_pca, train_y = get_important_features(train_x, train_y, best_feature_count)  

    n_splits = 5
    num_class = 8
    # best_params = get_best_params(x_pca, train_y, num_class) # 最適ハイパーパラメータの探索

    best_params = {
        'lambda_l1': 5.96,
        'lambda_l2': 1.1,
        'num_leaves': 12,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.89,
        'bagging_freq': 7,
        #'min_child_sample': 100
    }

    submission = np.zeros((len(test_x),num_class))
    accs = {}

    gkf = GroupKFold(n_splits=5)
    for i, (tr_idx, val_idx) in enumerate(gkf.split(train_x_resampled, train_y_resampled, groups=train_x_resampled["投手ID"])):
        tr_x = train_x_resampled.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y_resampled.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x_resampled.iloc[val_idx].reset_index(drop=True)
        val_y = train_y_resampled.iloc[val_idx].reset_index(drop=True)

        tr_weight = [

        ]
        tr_dataset = lgb.Dataset(tr_x, tr_y, free_raw_data=False)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset, free_raw_data=False)
        model, evals_result = get_model(tr_dataset, val_dataset, num_class, best_params)
        
        # 学習曲線の描画
        fig = lgb.plot_metric(evals_result, metric="multi_logloss")
        plt.savefig(f"{DATA_DIR}/learning_curve_{i}.png")

        y_pred = np.argmax(model.predict(val_x), axis=1) # 0~8の確率
        acc = accuracy_score(val_y, y_pred)
        accs[i] = acc
        print("#################################")
        print(f"accuracy: {acc}")
        print("#################################")
        y_preda = model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率
        submission += y_preda

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print(accs)
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/my_submission33.csv", header=False)


if __name__ == "__main__":
    main()