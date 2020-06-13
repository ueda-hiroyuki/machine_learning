import joblib
import logging
import optuna #ハイパーパラメータチューニング自動化ライブラリ
import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t 
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from functools import partial
from sklearn.feature_selection import RFECV, RFE
from optuna.integration import lightgbm_tuner #LightGBM用Stepwise Tuningに必要
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from python_file.kaggle.common import common_funcs as cf

"""
「社員の退職予測」
・カテゴリ変数名：sales, salary
・ヘッダー項目（以下11項目、以下順番で構成）
　・index
　・left (ラベル)
　・satisfaction_level
　・last_evaluation
　・number_project
　・average_montly_hours
　・time_spend_company
　・Work_accident
　・promotion_last_5years
　・sales
　・salary

"""

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/AI_jobcolle/final_tasks"
TRAIN_DATA_PATH = f"{DATA_DIR}/final_hr_analysis_train.csv"
TEST_DATA_PATH = f"{DATA_DIR}/final_hr_analysis_test.csv"
SAVE_MODEL_PATH = f"{DATA_DIR}/model.pkl"
SAVE_DATA_PATH = f"{DATA_DIR}/submission.csv"


def get_best_params(train_x: t.Any, train_y: t.Any) -> t.Any:
    tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y)
    best_params = {}
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt', 
    }
    best_params = {}
    tuning_history = []
    gbm = lightgbm_tuner.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,
        early_stopping_rounds=20,
        verbose_eval=50,
        best_params=best_params,
        tuning_history=tuning_history
    )
    return best_params


def get_model(tr_dataset: t.Any, val_dataset: t.Any, params: t.Dict[str, t.Any]) -> t.Any:
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=1000,
    )
    return model


def objective(X, y, trial):
    """最適化する目的関数"""
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, random_state=1)
    gbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type= 'gbdt', 
        n_jobs = 4,
        n_estimators=1000,
    )
    # RFE で取り出す特徴量の数を最適化する
    n_features_to_select = trial.suggest_int('n_features_to_select', 1, len(list(tr_x.columns))),
    rfe = RFE(estimator=gbm, n_features_to_select=n_features_to_select)
    rfe.fit(tr_x, tr_y)
    selected_cols = list(tr_x.columns[rfe.support_])
    
    tr_x_selected = tr_x.loc[:, selected_cols]
    val_x_selected = val_x.loc[:, selected_cols]
    gbm.fit(
        tr_x_selected, 
        tr_y,
        eval_set=[(val_x_selected, val_y)],
        early_stopping_rounds=20
    )
    y_pred = gbm.predict(val_x_selected)
    f1 = f1_score(val_y, y_pred, average="micro")
    return f1

def get_important_features(train_x: t.Any, train_y: t.Any, best_feature_count: int):
    gbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type= 'gbdt', 
        n_jobs = 4,
    )
    selector = RFE(gbm, n_features_to_select=best_feature_count)
    selector.fit(train_x, train_y) # 学習データを渡す
    selected_train_x = pd.DataFrame(selector.transform(train_x), columns=train_x.columns[selector.support_])
    return selected_train_x, train_y



def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH) 
    test_df = pd.read_csv(TEST_DATA_PATH)

    train_df["usage"] = "train"
    test_df["usage"] = "test"
    test_df["left"] = 100

    df = pd.concat([train_df, test_df], axis=0)
    usage = df.loc[:, "usage"]
    label = df.loc[:, "left"]
    df = df.drop(["usage", "left"], axis=1)

    categorical_columns = [c for c in df.columns if df[c].dtype == 'object']
    ce_ohe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_df = ce_ohe.fit_transform(df) 
    encorded_df = pd.concat([encorded_df, usage, label], axis=1)

    train = encorded_df[encorded_df["usage"] == "train"].drop("usage", axis=1).reset_index(drop=True)
    test = encorded_df[encorded_df["usage"] == "test"].drop("usage", axis=1).reset_index(drop=True)

    train_x = train.drop(["left", "index"], axis=1)
    train_y = train.loc[:,"left"]
    index = test.loc[:,"index"]
    test_x = test.drop(["left", "index"], axis=1)

    f = partial(objective, train_x, train_y) # 目的関数に引数を固定しておく
    study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する

    study.optimize(f, n_trials=10) # 試行回数を決定する
    print('params:', study.best_params)# 発見したパラメータを出力する
    best_feature_count = study.best_params['n_features_to_select']
    train_x, train_y = get_important_features(train_x, train_y, best_feature_count)  

    n_splits = 10
    best_params = get_best_params(train_x, train_y)


    submission = np.zeros((len(test_x),1))
    acc_scores = {}


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(skf.split(train_x, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset, best_params)

        y_pred = model.predict(test_x)
        preds = pd.DataFrame(y_pred)
        submission += preds

    submission_df = pd.DataFrame(submission/n_splits)
    
    submission_df = pd.concat([index, submission_df], axis=1)
    print("#################################")
    print(submission_df)
    print("#################################")


    submission_df.to_csv(SAVE_DATA_PATH, header=False, index=False)



if __name__ == "__main__":
    main()