import gc
import logging
import typing as t
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import category_encoders as ce
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.feature_selection import RFE, RFECV
from datetime import datetime
from more_itertools import windowed
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score


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
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
PITCH_REMOVAL_COLUMNS = ["時刻"]
PLAYER_REMOVAL_COLUMNS = ["出身地", "出身国", "出身高校名", "出身大学名", "生年月日", "社会人", "ドラフト年", "ドラフト種別", "ドラフト順位", "年俸", "背番号"]

PRE_TRAIN_PARAMS = {
    "objective": 'multiclass',
    "boosting_type": 'gbdt',
    "metric": 'multi_logloss',
    'num_class':  8,
    'learning_rate': 0.2,
    'n_estimators': 10000,
    'min_data_in_leaf': 1000,
    'num_leaves': 20,
    'num_iterations' : 100,
    'feature_fraction' : 0.7,
    'max_depth' : 10
}

NUM_CLASS = 8


def get_important_cols(pre_train_x, pre_train_y, pre_valid_x, pre_valid_y, pre_train_params=PRE_TRAIN_PARAMS):    
    pre_train_dataset = lgb.Dataset(pre_train_x, pre_train_y)
    pre_valid_dataset = lgb.Dataset(pre_valid_x, pre_valid_y, reference=pre_train_dataset)
    model = lgb.train(
        params=pre_train_params,
        train_set=pre_train_dataset,
        valid_sets=pre_valid_dataset,
        early_stopping_rounds=5,
    )
    pre_importance_df = pd.DataFrame({
        'feature': pre_train_x.columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    for i in pre_importance_df.items():
        print(i)
    
    important_feature = pre_importance_df.iloc[0:30]
    important_cols = list(important_feature["feature"])
    return important_cols


def get_selected_data(train_x, train_y):
    est = lgb.LGBMClassifier()
    selector = RFECV(estimator=est, step=0.05, n_jobs=1, min_features_to_select=100, cv=skf, verbose=10)



def get_best_params(df, num_class) -> t.Any:
    _train = df[df["月"] < 9]
    _valid = df[df["月"] >= 9]
    tr_x = _train.drop(["球種", "年度"], axis=1)
    val_x = _valid.drop(["球種", "年度"], axis=1)
    tr_y = _train.loc[:, "球種"]
    val_y = _valid.loc[:, "球種"]

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
        num_boost_round=100,
        early_stopping_rounds=5,
        verbose_eval=10,
        best_params=best_params,
        tuning_history=tuning_history
    )

    return best_params


def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=val_dataset,
        early_stopping_rounds=5,
        verbose_eval=10
    )
    return model


def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    train_pitch["月"] = pd.to_datetime(train_pitch['日付']).dt.month
    test_pitch["月"] = pd.to_datetime(test_pitch['日付']).dt.month
    test_pitch["球種"] = 0
    test_pitch["投球位置区域"] = 0

    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)
    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged_data = pd.merge(
        pitch_data, 
        player_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域'], axis=1).fillna(0)
    label = merged_data.loc[:, "球種"]
    merged_data = merged_data.drop("球種", axis=1)
    print(merged_data["日付"].value_counts())

    grouped = merged_data.groupby("日付")
    print(grouped.shape)
    
    train_x, valid_x, train_y, valid_y = train_test_split(grouped, label, test_size=0.2)
    
    print(train_x["日付"].value_counts())
    print(valid_x["日付"].value_counts())


    # categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == "object"]
    # ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    # ce_oe.fit(merged_data)
    # encorded_data = ce_oe.fit_transform(merged_data)
    # encorded_data = pd.concat([encorded_data, label], axis=1)
    
    # train = encorded_data[encorded_data["年度"] == 2017] # 2017年度データは学習用

    
    
    
    # train_x = encorded_data.drop("球種", axis=1)
    # train_y = encorded_data.loc[:, "球種"]
    # selected_train_x, train_y = get_selected_data(train_x, train_y)
    

    




    # pre_train_data = merged_data[merged_data["年度"] == 2017] # 2017年度データは学習用
    # pre_train_data = cf.label_encorder(pre_train_data)
    # pre_train = pre_train_data[pre_train_data["月"] < 9] # 2017/9以前のデータ(3,4,5,6,7,8)
    # pre_valid =  pre_train_data[pre_train_data["月"] >= 9] # 2017/9以降のデータ(9,10)
    # years = merged_data.loc[:, "年度"]

    # pre_train_x = pre_train.drop(["球種","月","年度", "日付"], axis=1)
    # pre_train_y = pre_train.loc[:, "球種"]
    # pre_valid_x = pre_valid.drop(["球種","月","年度", "日付"], axis=1)
    # pre_valid_y = pre_valid.loc[:, "球種"]

    # important_cols = get_important_cols(pre_train_x, pre_train_y, pre_valid_x, pre_valid_y)
    # dataset = merged_data.loc[:, [*important_cols, "球種", "月"]]
    # dataset = cf.label_encorder(dataset)
    # dataset = pd.concat([dataset, years], axis=1)
    # cf.check_corr(dataset, "predict_pitching_type")

    # train = dataset[dataset["年度"] == 2017]
    # test = dataset[dataset["年度"] != 2017]

    # train_x = train.drop(["球種", "年度"], axis=1)
    # train_y = train.loc[:, "球種"]
    # test_x = test.drop(["球種", "年度"], axis=1).reset_index(drop=True)
    # n_splits = 5

    # best_params = get_best_params(train, NUM_CLASS) # 最適ハイパーパラメータの探索   
    # # best_params = {
    # #     "objective": 'multiclass',
    # #     "boosting_type": 'gbdt',
    # #     "metric": 'multi_logloss',
    # #     'num_class':  NUM_CLASS,
    # #     'learning_rate': 0.1,
    # #     'n_estimators': 100,
    # #     'min_data_in_leaf': 2000,
    # #     'num_leaves': 10,
    # #     'num_iterations' : 100,
    # #     'feature_fraction' : 0.7,
    # #     'max_depth' : 10
    # # }

    # submission = np.zeros((len(test_x),NUM_CLASS))
    # kf = list(windowed(range(3,11), 4, step=1))
    # for months in kf:
    #     _train = train[
    #         (train["月"] == months[0]) |
    #         (train["月"] == months[1]) |
    #         (train["月"] == months[2])
    #     ]
    #     _train_x = _train.drop(["球種", "年度"], axis=1).reset_index(drop=True)
    #     _train_y = _train.loc[:, "球種"].reset_index(drop=True)

    #     _valid = train[train["月"] == months[-1]]
    #     _valid_x = _valid.drop(["球種", "年度"], axis=1).reset_index(drop=True)
    #     _valid_y = _valid.loc[:, "球種"].reset_index(drop=True)
        
    #     tr_dataset = lgb.Dataset(_train_x, _train_y)
    #     val_dataset = lgb.Dataset(_valid_x, _valid_y, reference=tr_dataset)
    #     model = get_model(tr_dataset, val_dataset, best_params)
    #     y_pred = model.predict(test_x, num_iteration=model.best_iteration)
    #     submission += y_pred

    # submission_df = pd.DataFrame(submission/n_splits)
    # print("#################################")
    # print(submission_df)
    # print(best_params) 
    # print("#################################")
    
    # submission_df.to_csv(f"{DATA_DIR}/my_submission21.csv", header=False)


if __name__ == "__main__":
    main()