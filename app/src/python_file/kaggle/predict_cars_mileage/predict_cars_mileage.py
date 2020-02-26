import pandas as pd
import numpy as np
import lightgbm as lgb
import typing as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def change_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('id', axis=1)
    df = df.replace('?', None)
    df = df.drop('car name', axis=1).astype({'cylinders': float, 'horsepower': float, 'model year': float, 'origin': float})
    return df

def process_train(path: str) -> t.Tuple[pd.DataFrame, pd.Series]:
    train = pd.read_table(path)
    name_df = train.loc[:,'car name']
    train = change_to_float(train)
    return train, name_df

def process_test(path: str) -> t.Tuple[pd.DataFrame, pd.Series]:
    test = pd.read_table(path)
    id_df = test.loc[:,'id']
    test = change_to_float(test)
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
    plt.figure(figsize=(20,5))
    df.plot()
    plt.savefig('src/sample_data/Kaggle/predict_cars_mileage/feature_figure.png')

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
        early_stopping_rounds=5,
    )
    return model


def main() -> None:    
    train, name_df = process_train('src/sample_data/Kaggle/predict_cars_mileage/train.tsv')
    test, id_df = process_test('src/sample_data/Kaggle/predict_cars_mileage/test.tsv')
    train_x = train.drop('mpg', axis=1)
    train_y = train.loc[:,'mpg']
    train_x = standardize(train_x)

    check_fig(train_x)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    models = []
    for tr_idx, val_idx in kf.split(train_x, train_y):
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        val_x = train_x.iloc[val_idx].reset_index(drop=True)
        val_y = train_y.iloc[val_idx].reset_index(drop=True)
        print(tr_x, tr_y, val_x, val_y)
        tr_dataset = lgb.Dataset(tr_x, tr_y)
        val_dataset = lgb.Dataset(val_x, val_y, reference=tr_dataset)
        model = get_model(tr_dataset, val_dataset)
        models.append(model)

    y_preds = []
    for model in models:
        y_pred = model.predict(test, num_iteration=model.best_iteration)
        print(y_pred)
        y_preds.append(y_pred)

    pred_df = pd.DataFrame(y_preds).transpose().mean(axis='columns')
    submission_df = pd.concat([id_df, pred_df], axis=1)
    print(submission_df)
    submission_df.to_csv('src/sample_data/Kaggle/predict_cars_mileage/submission.csv', header=False, index=False)



if __name__ == "__main__":
    main()