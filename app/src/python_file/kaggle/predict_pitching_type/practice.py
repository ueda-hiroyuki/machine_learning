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
import featuretools as ft
import category_encoders as ce # カテゴリ変数encording用ライブラリ
import optuna #ハイパーパラメータチューニング自動化ライブラリ
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.decomposition import PCA
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
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

# GRID_SEARCH_PARAMS = {
#     'knn':{
#         'est__n_neighbors':[2,3,4],
#         'est__weights':['uniform','distance'],
#         'est__algorithm':['auto'],
#         'est__leaf_size':[10,100],
#         'est__p':[1,2]
#     },
#     'logistic': {
#         "est__C":[0.1, 0.2, 0.5,  1],
#         "est__penalty":['l1', 'l2'],
#         'est__class_weight':['balanced'],
#         'est__max_iter':[1000, 2000]
#     },
#     'tree':{
#         'est__max_leaf_nodes': [10],
#         'est__min_samples_split': [5, 10],
#         'est__max_depth': [5, 10],
#         'est__criterion': ['gini', 'entropy'],
#         'est__class_weight':['balanced', None]
#     },
#     'rf':{
#         'est__min_samples_split':[5, 10],
#         'est__min_samples_leaf':[5, 10],
#         'est__max_depth': [5, 8],
#         "est__criterion": ["entropy"],
#         'est__class_weight':['balanced', None]
#     },
#     'gb':{
#         'est__loss':['deviance','exponential'],
#         'est__learning_rate':[0.01, 0.1],
#         'est__max_depth':[5, 10],
#         'est__min_samples_split':[0.1, 0.5],
#         'est__min_samples_leaf':[3, 5],
#     },
#     'SVC':{
#         "est__C":[0.1, 0.2, 0.5,  1],
#         'est__class_weight':['balanced'],
#         'est__max_iter':[1000, 2000]
#     },
#     'mlp':{
#         "est__hidden_layer_sizes":[(10,10), (10,10,10), (10,10,10), (10,10,10,10)],
#         "est__alpha":[0.1, 0.2, 0.5],
#         'est__early_stopping':[True],
#         'est__max_iter':[1000, 2000]
#     },
#     'adb':{
#         'est__n_estimators':[1000, 2000],
#         'est__learning_rate':[0.01, 0.1, 0.2]
#     }
# }
GRID_SEARCH_PARAMS = {
    'knn':{
        'est__n_neighbors':[2,3,4],
    },
    'logistic': {
        "est__C":[0.1, 0.2, 0.5,  1],
    },
    'tree':{
        'est__max_depth': [5, 10],
    },
    'rf':{
        'est__min_samples_split':[5, 10],
    },
    'gb':{
        'est__learning_rate':[0.01, 0.1],
    },
    'SVC':{
        "est__C":[0.1, 0.2, 0.5,  1],
    },
    'mlp':{
        "est__hidden_layer_sizes":[(10,10), (10,10,10), (10,10,10), (10,10,10,10)],
    },
    'adb':{
        'est__n_estimators':[100, 200],
    }
}


def main():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target)

    X, test_x, y, test_y = train_test_split(data, target, test_size=0.2, stratify=target) 
    best_estimetors = {}
    model_names = [c for c in PIPELINES]

    for (param_name, param), (pipeline_name, pipeline) in zip(GRID_SEARCH_PARAMS.items(), PIPELINES.items()):
        logging.info(f'{param_name} GRID SEARCH STARTED !!')
        gscv = GridSearchCV(pipeline, param, cv=5, refit=True)
        gscv.fit(X, y)
        best_estimetor = gscv.best_estimator_
        best_estimetors[pipeline_name] = best_estimetor

    meta_model = LogisticRegression() # meta_modelは線形モデル

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    for train_idx, val_idx in skf.split(X, y):
        tr_x = X.iloc[train_idx].reset_index(drop=True)
        tr_y = y.iloc[train_idx].reset_index(drop=True)
        val_x = X.iloc[val_idx].reset_index(drop=True)
        val_y = y.iloc[val_idx].reset_index(drop=True)

        train_x_base, valid_x_base, train_y_base, valid_y_base = train_test_split(tr_x, tr_y, test_size=0.2, stratify=tr_y)
        base_y_preds = []
        valid_y_preds = []
        test_y_preds = []
        for model_name in model_names:
            base_clf = best_estimetors[model_name]
            base_clf.fit(train_x_base, train_y_base)
            base_y_pred = base_clf.predict(valid_x_base)
            base_y_preds.append(base_y_pred)

            valid_y_pred = base_clf.predict(val_x)
            valid_y_preds.append(valid_y_pred)
            test_y_pred = base_clf.predict(test_x)
            test_y_preds.append(test_y_pred)

        base_preds = pd.DataFrame(base_y_preds).T
        valid_preds = pd.DataFrame(valid_y_preds).T
        test_preds = pd.DataFrame(test_y_preds).T

        meta_model.fit(base_preds, valid_y_base)
        meta_y_pred = meta_model.predict(valid_preds)
        test_y_pred = meta_model.predict(test_preds)
        valid_accuracy = accuracy_score(meta_y_pred, val_y)
        test_accuracy = accuracy_score(test_y_pred, test_y)

        print("#########################################")
        print(f"valid_accuracy is {valid_accuracy} !!")
        print(f"test_accuracy is {test_accuracy} !!")

           
    
if __name__ == "__main__":
    main()