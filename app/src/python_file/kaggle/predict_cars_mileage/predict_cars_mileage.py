import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def change_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('id', axis=1)
    df = df.replace('?', None)
    df = df.drop('car name', axis=1).astype({'cylinders': float, 'horsepower': float, 'model year': float, 'origin': float})
    return df

def process_train(path: str) -> t.Tuple[pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_table(path)
    print(train['car name'].value_counts())
    name_df = train.loc[:,'car name']
    train = change_to_float(train)
    train_y = train.loc[:,'mpg']
    train_x = train.drop(['mpg','cylinders'], axis=1)
    return train_x, train_y, name_df

def process_test(path: str) -> t.Tuple[pd.DataFrame, pd.Series]:
    test = pd.read_table(path)
    id_df = test.loc[:,'id']
    test = change_to_float(test)
    test = test.drop('cylinders', axis=1)
    return test, id_df

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df)
    scaler_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return scaler_df

def minmaxscaler(df: pd.DataFrame) -> pd.DataFrame:
    mm = MinMaxScaler()
    mm.fit(df)
    mm_df = pd.DataFrame(mm.transform(df), columns=df.columns)
    return mm_df

def check_fig(df: pd.DataFrame) -> pd.DataFrame:
    for name, item in df.iteritems():
        plt.figure()
        item.plot()
        plt.savefig(f'src/sample_data/Kaggle/predict_cars_mileage/{name}.png')


def check_corr(df: pd.DataFrame) -> None:
    corr = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, square=True, annot=True)
    plt.savefig(f'src/sample_data/Kaggle/predict_cars_mileage/corr_heatmap.png')


def get_model(tr_dataset: t.Any, val_dataset: t.Any) -> t.Any:
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        'metric' : {'l2'},
        'num_leaves' : 5,
        'min_data_in_leaf': 5,
        'num_iterations' : 50,
        'learning_rate' : 0.2,
        'feature_fraction' : 0.5,
    }
    model = lgb.train(
        params=params,
        train_set=tr_dataset,
        valid_sets=val_dataset,
        early_stopping_rounds=10,
    )
    return model


def main() -> None:    
    train_x, train_y, name_df = process_train('src/sample_data/Kaggle/predict_cars_mileage/train.tsv')
    test, id_df = process_test('src/sample_data/Kaggle/predict_cars_mileage/test.tsv')
    
    check_corr(train_x)
    check_fig(train_x)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    models = []
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)
        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset)
        models.append(model)

    y_preds = []
    importances = []
    for model in models:
        importance = pd.DataFrame(model.feature_importance(), index=train_x.columns, columns=["importance"])
        importances.append(importance)
        y_pred = model.predict(test, num_iteration=model.best_iteration)
        y_preds.append(pd.Series(y_pred))
    preds_df = pd.concat(y_preds, axis=1)
    pred_df = preds_df.mean(axis=1)
    submission_df = pd.concat([id_df, pred_df], axis=1)
    print(pd.concat(importances, axis=1))
    print(submission_df)
    submission_df.to_csv('src/sample_data/Kaggle/predict_cars_mileage/submission.csv', header=False, index=False)



if __name__ == "__main__":
    main()