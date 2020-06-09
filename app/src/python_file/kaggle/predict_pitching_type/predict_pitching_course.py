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
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from python_file.kaggle.common import common_funcs as cf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score


"""
train_pitch(51 columns)
test_pitch(49 columns)
2つの差は「球種」と「投球位置区域」である
⇒「投球位置区域」0~12の13種類分類

train_player(25 columns)
test_player(25 columns)

・pitchとplayerで共通しているcolumn：「年度」,
・submission.csvに記載する情報は「test_pitchのデータ連番」「各コース(13種)の投球確率」

"""

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/predict_pitching_type"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
PITCH_REMOVAL_COLUMNS = [
    "日付", 
    "時刻", 
    "年度", 
    "試合内連番", 
    "成績対象打者ID", 
    "成績対象投手ID", 
    "打者試合内打席数",
    "投手役割",
    "イニング",
    # "試合ID",
    "表裏",
    # "一塁走者ID",
    # "二塁走者ID",
    # "三塁走者ID",
    # "球場名",
    "プレイ前ホームチーム得点数",
    "プレイ前アウェイチーム得点数",
    # "アウェイチームID",
    # "球場ID",
    # "打者チームID",
    "試合種別詳細",
    # "プレイ前アウト数",
    "ホームチームID",
    # "左翼手ID",
    # "データ内連番",
    "投手登板順",
    # "打者ID",
    # "打者打順",
    # "試合内投球数",
    # "一塁手ID",
    # "二塁手ID",
    # "投手イニング内投球数",
    # "イニング内打席数",
]
PLAYER_REMOVAL_COLUMNS = [
    "出身高校名", 
    "出身大学名", 
    "生年月日", 
    "位置", 
    "出身地", 
    "出身国", 
    "年度", 
    "チームID", 
    "社会人",
    "育成選手F",
    "投",
    "打",
    "ドラフト種別",
    "ドラフト順位",
    "血液型",
]
LABEL_ENCORDER_COLUMNS = [
    "球場名", 
    "試合種別詳細", 
    "表裏", 
    "投手投球左右", 
    "投手役割", 
    "打者打席左右", 
    "打者守備位置", 
    "プレイ前走者状況", 
    "チーム名", 
    "選手名", 
    "投", 
    "打", 
    "血液型"
]

def remove_columns(df):
    df = df.drop(REMOVAL_COLUMNS, axis=1).fillna(0)
    return df


def label_encorder(col: pd.Series) -> pd.DataFrame:
    if col.dtypes == "object":
        encorded_col = LabelEncoder().fit_transform(col)
        print(encorded_col)
        return encorded_col
    else:
        print(col)
        return col


def get_best_params(train_x: t.Any, train_y: t.Any, num_class: int) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
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
        num_boost_round=500,
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
        num_boost_round=500,
    )
    return model


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


def get_important_features(train_x: t.Any, test_x: t.Any, best_feature_count: int):
    pca = PCA(n_components=best_feature_count).fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)
    return train_x_pca, test_x_pca



def main():
    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)
    sub = pd.read_csv(SUBMISSION_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test" 
    pitch_data = pd.concat([train_pitch, test_pitch], axis=0).drop(PITCH_REMOVAL_COLUMNS, axis=1)

    player_data = pd.concat([train_player, test_player], axis=0).drop(PLAYER_REMOVAL_COLUMNS, axis=1) #.fillna(0)
    pitchers_data = train_player[train_player["位置"] == "投手"].drop(PLAYER_REMOVAL_COLUMNS, axis=1)

    merged_data = pd.merge(
        pitch_data, 
        pitchers_data, 
        how="left", 
        left_on='投手ID', 
        right_on='選手ID'
    ).drop(['選手ID', '球種'], axis=1)

    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use", "位置", "年度"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_data) 
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    train_x = train.drop("投球位置区域", axis=1)
    train_y = train.loc[:,"投球位置区域"].astype(int)
    test_x = test.drop("投球位置区域", axis=1).reset_index(drop=True)


    f = partial(objective, train_x, train_y) # 目的関数に引数を固定しておく
    study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する

    study.optimize(f, n_trials=10) # 試行回数を決定する
    print('params:', study.best_params)# 発見したパラメータを出力する
    best_feature_count = study.best_params['n_components']
    train_x_pca, test_x_pca = get_important_features(train_x, train_y, best_feature_count)  

    n_splits = 10
    num_class = 13
    best_params = get_best_params(train_x_pca, train_y, num_class) # 最適ハイパーパラメータの探索
    
    submission = np.zeros((len(test_x_pca),num_class))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(train_x_pca, train_y):
        tr_x = train_x_pca.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x_pca.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset, best_params)
        y_pred = model.predict(test_x_pca, num_iteration=model.best_iteration)
        submission += y_pred
        

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################submission & best_params################")
    print(submission_df)
    print(best_params) 
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/submission_pitching_course5   .csv", header=False)


if __name__ == "__main__":
    main()