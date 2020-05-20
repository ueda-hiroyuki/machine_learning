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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


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
# print(mpl.matplotlib_fname())
# print(mpl.get_configdir())
# print(mpl.get_cachedir())

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/predict_pitching_type"
TRAIN_PITCH_PATH = f"{DATA_DIR}/train_pitch.csv"
TRAIN_PLAYER_PATH = f"{DATA_DIR}/train_player.csv"
TEST_PITCH_PATH = f"{DATA_DIR}/test_pitch.csv"
TEST_PLAYER_PATH = f"{DATA_DIR}/test_player.csv"
SUBMISSION_PATH = f"{DATA_DIR}/sample_submit_ball_type.csv"
PITCH_REMOVAL_COLUMNS = ["日付", "時刻", "試合内連番", "成績対象打者ID", "成績対象投手ID", "打者試合内打席数", "試合ID", "試合内投球数"]
PLAYER_REMOVAL_COLUMNS = ["出身高校名", "出身大学名", "生年月日", "位置", "出身地", "出身国", "チームID", "社会人","ドラフト年","ドラフト種別","ドラフト順位", "年俸", "育成選手F"]


def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    y_true = data.get_label()
    y_preds = np.reshape(preds, [len(y_true), 8], order='F')
    y_pred = np.argmax(y_preds, axis=1)
    metric = np.mean(y_true == y_pred)
    return 'accuracy', metric, True

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
        num_boost_round=1000,
        early_stopping_rounds=5,
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
        #early_stopping_rounds=5,
        num_boost_round=500,
        valid_names=['eval','train'],
        evals_result=evals_result,
        feval=accuracy,
    )
    return model, evals_result


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
        player_data, 
        how="left", 
        left_on=['年度','投手ID'], 
        right_on=['年度','選手ID'],
    ).drop(['選手ID', '投球位置区域', "年度"], axis=1).fillna(0)
    use = merged_data.loc[:, "use"]
    merged_data = merged_data.drop(["use"], axis=1)
    encorded_data = cf.label_encorder(merged_data)
    # encorded_data = cf.standardize(encorded_data) # 標準化
    encorded_data = pd.concat([encorded_data, use], axis=1)
 
    train = encorded_data[encorded_data["use"] == "train"].drop("use", axis=1)
    test = encorded_data[encorded_data["use"] == "test"].drop("use", axis=1)

    train_x = train.drop("球種", axis=1)
    train_y = train.loc[:,"球種"]
    test_x = test.drop("球種", axis=1).reset_index(drop=True)

    # cf.check_corr(train_x, "predict_pitching_type")

    n_splits = 10
    num_class = 8
    # best_params = get_best_params(train_x, train_y, num_class) # 最適ハイパーパラメータの探索
    best_params = {
        "objective": 'multiclass',
        "boosting_type": 'gbdt',
        "metric": 'multi_logloss',
        'num_class':  num_class,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_data_in_leaf': 2000,
        'num_leaves': 10,
        'num_iterations' : 1000,
        'feature_fraction' : 0.7,
        'max_depth' : 10
    }
    submission = np.zeros((len(test_x),num_class))
    accs = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
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
        _, ax1 = plt.subplots()
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
        print(f"accuracy_score: {acc}")
        print("#################################")
        y_preda = model.predict(test_x, num_iteration=model.best_iteration) # 0~8の確率
        submission += y_preda

    submission_df = pd.DataFrame(submission/n_splits)
    print("#################################")
    print(submission_df)
    print(best_params) 
    print(accs)
    print("#################################")
    
    submission_df.to_csv(f"{DATA_DIR}/my_submission15.csv", header=False)


if __name__ == "__main__":
    main()
