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
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from functools import partial
from python_file.kaggle.common import common_funcs as cf
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
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
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"

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
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'num_class': NUM_CLASS,
        **params
    }
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=[val_dataset, tr_dataset],
        early_stopping_rounds=20,
        num_boost_round=10000,
        valid_names=['eval','train'],
        evals_result=evals_result,
        verbose_eval=50,
        feval=accuracy,
    )
    return model, evals_result

def objective(X, y, trial):
    """最適化する目的関数"""
    n_components = trial.suggest_int('n_components', 1, len(list(X.columns))),
    pca = PCA(n_components=n_components[0]).fit(X)
    x_pca = pca.transform(X)
    tr_x, val_x, tr_y, val_y = train_test_split(x_pca, y, test_size=0.2, random_state=1, stratify=y)
    train_set = lgb.Dataset(tr_x, label=tr_y)
    valid_set = lgb.Dataset(val_x, label=val_y, reference=train_set)
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'num_class': NUM_CLASS,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 1.0),
        'max_depth': trial.suggest_int('max_depth', 5, 100),
    }
    gbm = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=valid_set,
        early_stopping_rounds=20,
        verbose_eval=50,
        num_boost_round=10000,
    )
    y_preda = gbm.predict(val_x)
    y_pred = np.argmax(y_preda, axis=1) # 0~8の確率
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
    sub = pd.read_csv(SUBMISSION_PATH)

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

    labal = merged_data.loc[:, "球種"]
    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use", "位置", "年度", "球種"], axis=1)

    # category_encodersによってカテゴリ変数をencordingする
    categorical_columns = [c for c in merged_data.columns if merged_data[c].dtype == 'object']
    ce_oe = ce.OrdinalEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_data = ce_oe.fit_transform(merged_data) 
    encorded_data = pd.concat([encorded_data, use], axis=1)

    rus = RandomUnderSampler(random_state=1)
    X, y = rus.fit_resample(encorded_data, labal)
    new_use = X.loc[:, "use"]
    X = X.drop(["use"], axis=1)

    x_reduced = TSNE(n_components=2, random_state=1).fit_transform(X)
    x_reduced_df = pd.DataFrame(x_reduced, columns=["tnse_x", "tnse_y"])

    dataset = pd.concat([X, x_reduced_df], axis=1)
    dataset = cf.standardize(dataset) # 標準化
    dataset = pd.concat([dataset, new_use, labal], axis=1)

    train = dataset[dataset["use"] == "train"].drop("use", axis=1).reset_index(drop=True)
    test = dataset[dataset["use"] == "test"].drop("use", axis=1).reset_index(drop=True)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1)

    print(train_x, train_y, test_x)
    

    f = partial(objective, train_x, train_y) # 目的関数に引数を固定しておく
    study = optuna.create_study(direction='maximize') # Optuna で取り出す特徴量の数を最適化する

    study.optimize(f, n_trials=20) # 試行回数を決定する
    study_result = study.best_params
    best_feature_count = study_result.pop('n_components')
    best_params = study_result
    x_pca, train_y = get_important_features(train_x, train_y, best_feature_count)  
    # best_params = get_best_params(x_pca, train_y, num_class) # 最適ハイパーパラメータの探索
    n_splits = 10
    submission = np.zeros((len(test_x), NUM_CLASS))
    accs = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for i, (tr_idx, val_idx) in enumerate(skf.split(x_pca, train_y)):
        tr_x = train_x.iloc[tr_idx].reset_index(drop=True)
        tr_y = train_y.iloc[tr_idx].reset_index(drop=True)
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)

        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model, evals_result = get_model(tr_dataset, val_dataset, best_params)
        
        # 学習曲線の描画
        eval_metric_logloss = evals_result['eval']['multi_logloss']
        train_metric_logloss = evals_result['train']['multi_logloss']
        eval_metric_acc = evals_result['eval']['accuracy']
        train_metric_acc = evals_result['train']['accuracy']
        _, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(eval_metric_logloss, label='eval logloss', c='r')
        ax1.plot(train_metric_logloss, label='train logloss', c='b')
        ax1.set_ylabel('logloss')
        ax1.set_xlabel('rounds')
        ax1.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.plot(eval_metric_acc, label='eval accuracy', c='g')
        ax2.plot(train_metric_acc, label='train accuracy', c='y')
        ax2.set_ylabel('accuracy')
        ax2.legend(loc='lower right')
        plt.savefig(f'{DATA_DIR}/learning_{i}.png')

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
    print(study.best_params)
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/my_submission19.csv", header=False)


if __name__ == "__main__":
    main()
