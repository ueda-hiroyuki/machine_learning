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
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
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
PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "年度", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数", "試合ID"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "位置", "出身地", "出身国", "年度", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸"]
LABEL_ENCORDER_COLUMNS = ["球場名", "試合種別詳細", "表裏", "投手投球左右", "投手役割", "打者打席左右", "打者守備位置", "プレイ前走者状況", "チーム名", "選手名", "投", "打", "血液型"]

def remove_columns(df):
    df = df.drop(REMOVAL_COLUMNS, axis=1).fillna(0)
    return df

def get_best_col_names(train_x: t.Any, train_y: t.Any, best_params: t.Any) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
    tr_dataset = lgb.Dataset(tr_x, tr_y)
    val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
    model = lgb.train(
        params=best_params,
        train_set=tr_dataset,
        valid_sets=val_dataset,
        early_stopping_rounds=5,
    )
    pre_importance_df = pd.DataFrame({
        'feature': tr_x.columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    important_feature = pre_importance_df.iloc[0:20]
    important_cols = list(important_feature["feature"])
    return important_cols



def get_best_params(train_x: t.Any, train_y: t.Any, num_class: int) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
    best_params = {}
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        n_jobs = 4,
    )
    grid_params = {
        'learning_rate': [0.1, 0.2, 0.5],
        'n_estimators': [10000],
        'min_data_in_leaf': [10, 100, 500, 1000, 1500, 2000],
        'num_leaves': [10, 20, 50],
        'num_iterations' : [100, 200, 500, 1000],
        'feature_fraction' : [0.7, 0.8],
        'max_depth' : [5, 10, 20, 50]
    }
    # grid_params = {
    #     'learning_rate': [0.1],
    #     'n_estimators': [50],
    #     'min_data_in_leaf': [2000],
    #     'num_leaves': [10],
    #     'num_iterations' : [10],
    #     'feature_fraction' : [0.7],
    #     'max_depth' : [10]
    # }
    _kf = KFold(n_splits=10, shuffle=True, random_state=0)
    grid_search = GridSearchCV(
        gbm, # 分類器,
        param_grid=grid_params, # 試したいパラメータの渡し方
        cv=_kf, # 5分割交差検証
    )
    grid_search.fit(
        tr_x,
        tr_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=5
    )

    best_params['learning_rate'] = grid_search.best_params_['learning_rate']
    best_params['min_data_in_leaf'] = grid_search.best_params_['min_data_in_leaf']
    best_params['n_estimators'] = grid_search.best_params_['n_estimators']
    best_params['num_leaves'] = grid_search.best_params_['num_leaves']
    best_params['num_iterations'] = grid_search.best_params_['num_iterations']
    best_params['feature_fraction'] = grid_search.best_params_['feature_fraction']
    best_params['max_depth'] = grid_search.best_params_['max_depth']
    best_params["objective"] = 'multiclass'
    best_params["boosting_type"] = 'gbdt'
    best_params["metric"] = 'multi_logloss'
    best_params['num_class'] = num_class
    return best_params

def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=val_dataset,
        early_stopping_rounds=5,
    )
    return model

def gen_r2_score_fig(score_list, n_splits, save_dir):
    plt.figure()
    plt.plot(range(1, n_splits+1), marker='o')
    plt.xlabel("epochs")
    plt.ylabel("r2_score")
    plt.xlim(0, n_splits+1)
    plt.savefig(f"{save_dir}/r2_score.png")


def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test"
    test_pitch["球種"] = 0
    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged_data = pd.merge(
        pitch_data, 
        pitchers_data, 
        how="left", 
        left_on='投手ID', 
        right_on='選手ID'
    ).drop(['選手ID', '投球位置区域'], axis=1)
    dropna_merged_data = merged_data.dropna(how="any").reset_index(drop=True)
    use = dropna_merged_data.loc[:, "use"]
    encorded_data = cf.label_encorder(dropna_merged_data.drop(["use"], axis=1))
    # cf.check_hist(encorded_data, "predict_pitching_type")
    # encorded_data = cf.standardize(encorded_data) # 標準化
    
    encorded_data = pd.concat([encorded_data, use], axis=1)
    
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1).reset_index(drop=True)

    n_splits = 5
    num_class = 8
    best_params = get_best_params(train_x, train_y, num_class) # 最適ハイパーパラメータの探索
    best_col_names = get_best_col_names(train_x, train_y, best_params)
    # best_params = {
    #     "objective": 'multiclass',
    #     "boosting_type": 'gbdt',
    #     "metric": 'multi_logloss',
    #     'num_class':  num_class,
    #     'learning_rate': 0.2,
    #     'n_estimators': 50,
    #     'min_data_in_leaf': 1000,
    #     'num_leaves': 20,
    #     'num_iterations' : 100,
    #     'feature_fraction' : 0.7,
    #     'max_depth' : 10
    # }
    important_feature_df = merged_data.loc[:, [*best_col_names, "use", "球種"]]
    use = important_feature_df.loc[:, "use"]
    important_feature_df = cf.label_encorder(important_feature_df.drop(["use"], axis=1))
   
    important_feature_df = pd.concat([important_feature_df, use], axis=1)
    train = important_feature_df[important_feature_df["use"] == "train"].drop("use", axis=1)
    test = important_feature_df[important_feature_df["use"] == "test"].drop("use", axis=1)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1).reset_index(drop=True)

    submission = np.zeros((len(test_x),num_class))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset, best_params)
        y_pred = model.predict(test_x, num_iteration=model.best_iteration)
        submission += y_pred

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/my_submission9.csv", header=False)


if __name__ == "__main__":
    main()