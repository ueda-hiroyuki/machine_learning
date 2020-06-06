import gc
import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.decomposition import PCA
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


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

PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数", "試合ID"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]

NUM_CLASS = 8

PIPELINES = {
    'knn': Pipeline([
        ('scl',StandardScaler()),
        ('est',KNeighborsClassifier())
    ]),
    'logistic': Pipeline([
        ('scl',StandardScaler()),
        ('est',LogisticRegression(random_state=1))
    ]),
    'tree': Pipeline([
        ('est',DecisionTreeClassifier(random_state=1))
    ]),
    'rf': Pipeline([
        ('est',RandomForestClassifier(random_state=1))
    ]),
    'gb': Pipeline([
        ('est',GradientBoostingClassifier(random_state=1))
    ]),
    'SVC': Pipeline([
        ('scl',StandardScaler()),
        ('est',SVC(random_state=1))
    ]),
    'mlp': Pipeline([
        ('scl',StandardScaler()),
        ('est',MLPClassifier(random_state=1))
    ]),
    'adb': Pipeline([
        ('est',AdaBoostClassifier(random_state=1))
    ]),
}

GRID_SEARCH_PARAMS = {
    'knn':{
        'est__n_neighbors':[10, 20, 50],
        'est__weights':['uniform','distance'],
        'est__algorithm':['auto'],
        'est__leaf_size':[10, 100],
        'est__p':[1, 2]
    },
    'logistic': {
        "est__C":[0.1, 0.2, 0.5,  1],
        "est__penalty":['l1', 'l2'],
        'est__class_weight':['balanced'],
        'est__max_iter':[1000, 2000]
    },
    'tree':{
        'est__max_leaf_nodes': [10, 100],
        'est__min_samples_split': [100, 200, 500],
        'est__max_depth': [5, 10],
        'est__criterion': ['gini', 'entropy'],
        'est__class_weight':['balanced']
    },
    'rf':{
        'est__min_samples_split':[5, 10],
        'est__min_samples_leaf':[100, 200, 500],
        'est__max_depth': [5, 8],
        "est__criterion": ["entropy"],
        'est__class_weight':['balanced']
    },
    'gb':{
        'est__loss':['deviance','exponential'],
        'est__learning_rate':[0.01, 0.1],
        'est__max_depth':[5, 10],
        'est__min_samples_split':[0.1, 0.5],
        'est__min_samples_leaf':[100, 200, 500],
    },
    'SVC':{
        "est__C":[0.1, 0.2, 0.5,  1],
        'est__class_weight':['balanced'],
        'est__max_iter':[1000, 2000]
    },
    'mlp':{
        "est__hidden_layer_sizes":[(10,10), (10,10,10), (10,10,10), (10,10,10,10)],
        "est__alpha":[0.1, 0.2, 0.5],
        'est__early_stopping':[True],
        'est__max_iter':[1000, 2000]
    },
    'adb':{
        'est__n_estimators':[1000, 2000],
        'est__learning_rate':[0.01, 0.1, 0.2]
    }
}
# GRID_SEARCH_PARAMS = {
#     'knn':{
#         'est__n_neighbors':[2,3,4],
#     },
#     'logistic': {
#         "est__C":[0.1, 0.2, 0.5,  1],
#     },
#     'tree':{
#         'est__max_depth': [5, 10],
#     },
#     'rf':{
#         'est__min_samples_split':[5, 10],
#     },
#     'gb':{
#         'est__learning_rate':[0.01, 0.1],
#     },
#     'SVC':{
#         "est__C":[0.1, 0.2, 0.5,  1],
#     },
#     'mlp':{
#         "est__hidden_layer_sizes":[(10,10), (10,10,10), (10,10,10), (10,10,10,10)],
#     },
#     'adb':{
#         'est__n_estimators':[100, 200],
#     }
# }

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

def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=10000,
        valid_names=['eval','train'],
        evals_result=evals_result,
        feval=accuracy,
    )
    return model, evals_result

def objective(X, y, trial):
    """最適化する目的関数"""
    logging.info("START OBJECTIVE !!")
    gbm = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type= 'gbdt', 
        n_jobs = 4,
        n_estimators=10000,
    )
    n_components = trial.suggest_int('n_components', 10, len(list(X.columns))),
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
    """
    << 処理の流れ >>
    データ読み込み ⇒ 投球データと選手データの結合(train,testも結合) ⇒ nanの置換 ⇒ カテゴリ変数の変換 ⇒
    RFEによる特徴量選択(個数の最適化) ⇒ ハイパーパラメータの最適化 ⇒ 交差検証
    """

    train_pitch = pd.read_csv(TRAIN_PITCH_PATH)
    train_player = pd.read_csv(TRAIN_PLAYER_PATH)
    test_pitch = pd.read_csv(TEST_PITCH_PATH)
    test_player = pd.read_csv(TEST_PLAYER_PATH)

    train_pitch["use"] = "train"
    test_pitch["use"] = "test"
    test_pitch["球種"] = 0
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

    # merged_data = merged_data.sample(n=5000, random_state=42).reset_index(drop=True)

    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use", "位置", "年度"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_data) 
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    f = partial(objective, train_x, train_y) # 目的関数に引数を固定しておく
    study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する

    study.optimize(f, n_trials=10) # 試行回数を決定する
    print('params:', study.best_params)# 発見したパラメータを出力する
    best_feature_count = study.best_params['n_components']
    train_x_pca, test_x_pca = get_important_features(train_x, test_x, best_feature_count)  

    # train_x_pca, test_x_pca = train_x, test_x
    
    best_estimetors = {}
    model_names = [c for c in PIPELINES]

    for (param_name, param), (pipeline_name, pipeline) in zip(GRID_SEARCH_PARAMS.items(), PIPELINES.items()):
        logging.info(f'{param_name} GRID SEARCH STARTED !!')
        gscv = GridSearchCV(pipeline, param, cv=5, refit=True)
        gscv.fit(train_x_pca, train_y)
        best_estimetor = gscv.best_estimator_
        best_estimetors[pipeline_name] = best_estimetor

    n_splits = 10
    meta_model = LogisticRegression() # meta_modelは線形モデル
    submission = np.zeros((len(test_x_pca),NUM_CLASS))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for idx, (train_idx, val_idx) in enumerate(skf.split(train_x_pca, train_y)):
        logging.info(f"START {idx}th split !!")
        tr_x = train_x_pca.iloc[train_idx].reset_index(drop=True)
        tr_y = train_y.iloc[train_idx].reset_index(drop=True)
        val_x = train_x_pca.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        train_x_base, valid_x_base, train_y_base, valid_y_base = train_test_split(tr_x, tr_y, test_size=0.2, stratify=tr_y)
        base_y_preds = []
        valid_y_preds = []
        test_y_preds = []
        for model_name in model_names:
            logging.info(f"Start Learning {model_name} model !!")
            base_clf = best_estimetors[model_name]
            base_clf.fit(train_x_base, train_y_base)
            base_y_pred = base_clf.predict(valid_x_base)
            base_y_preds.append(base_y_pred)

            valid_y_pred = base_clf.predict(val_x)
            valid_y_preds.append(valid_y_pred)
            test_y_pred = base_clf.predict(test_x_pca)
            test_y_preds.append(test_y_pred)

        base_preds = pd.DataFrame(base_y_preds).T
        valid_preds = pd.DataFrame(valid_y_preds).T
        test_preds = pd.DataFrame(test_y_preds).T

        meta_model.fit(base_preds, valid_y_base)
        meta_y_preda = meta_model.predict_proba(valid_preds)
        test_y_preda = meta_model.predict_proba(test_preds)
        logloss = log_loss(val_y, meta_y_preda)
        print("#########################################")
        print(f"logloss is {logloss} !!")
        submission += pd.DataFrame(test_y_preda)/n_splits

    submission.to_csv(f"{DATA_DIR}/my_submission21.csv", header=False)
    print(submission)



        


    
    


    







if __name__ == "__main__":
    main()