import joblib
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
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


def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH) 
    test_df = pd.read_csv(TEST_DATA_PATH)






if __name__ == "__main__":
    main()