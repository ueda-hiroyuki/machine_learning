
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score

"""
InvoiceNo    --請求書番号（レシートごとに振られる番号）
StockCode    --商品コード
Description  --商品名
Quantity     --購買数量
InvoiceDate  --購買日
UnitPrice    --商品単価
CustomerID   --顧客ID
Country      --国

"""

for_model_csv = "src/sample_data/AI_jobcolle/dm_for_model.csv" # 検証用(時系列歴に前)
for_fwd_csv = "src/sample_data/AI_jobcolle/dm_for_fwd.csv" # 検証用(時系列歴に後ろ)

def main():
    for_model_data = pd.read_csv(for_model_csv)
    for_fwd_data = pd.read_csv(for_fwd_csv)

    for_model_data["usage"] = "model"
    for_fwd_data["usage"] = "fwd"

    dataset = pd.concat([for_model_data, for_fwd_data], axis=0).reset_index(drop=True)
    usages = dataset.loc[:, "usage"]
    dataset = dataset.drop("usage", axis=1)

    # one-hot-encordng
    categorical_columns = [c for c in dataset.columns if dataset[c].dtype == 'object']
    ce_oe = ce.OneHotEncoder(cols=categorical_columns, handle_unknown='impute')
    encorded_dataset = ce_oe.fit_transform(dataset)
    
    # null impute
    imp = SimpleImputer()
    imp.fit(encorded_dataset)
    encorded_dataset = pd.DataFrame(imp.transform(encorded_dataset), columns=encorded_dataset.columns)
    encorded_dataset = pd.concat([encorded_dataset, usages], axis=1)

    # make data and label
    train_X = encorded_dataset[encorded_dataset["usage"] == "model"].drop(["usage", "tgt"], axis=1)
    train_y = encorded_dataset[encorded_dataset["usage"] == "model"].loc[:, "tgt"]
    test_X = encorded_dataset[encorded_dataset["usage"] == "fwd"].drop(["usage", "tgt"], axis=1)
    test_y = encorded_dataset[encorded_dataset["usage"] == "fwd"].loc[:, "tgt"]
    
    # feature selection 
    est = RandomForestClassifier(random_state=0)
    selector = RFECV(estimator=est, step=0.05)
    selector.fit(train_X, train_y)
    selected_columns = train_X.columns[selector.support_]
    selected_train_X = pd.DataFrame(selector.transform(train_X), columns=selected_columns)
    
    # define pipeline
    pipelines = {
        "Gradient Boosting": Pipeline([
            ('scl', StandardScaler()),
            ('est', GradientBoostingClassifier(random_state=1))
        ]), 
        "Random Forest": Pipeline([
            ('scl', StandardScaler()),
            ('est', RandomForestClassifier(random_state=1))
        ]),
        "LightGBM": Pipeline([
            ('scl', StandardScaler()),
            ('est', lgb.LGBMClassifier())
        ]),
        "Logistic Regression": Pipeline([
            ('scl', StandardScaler()),
            ('est', LogisticRegression())
        ])
    }
        
    for pipe_name, pipeline in pipelines.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        # learning & scoring(calcurate score while learning)
        pipeline.fit(selected_train_X, train_y) # learning
        results = cross_val_score(pipeline, selected_train_X, train_y, scoring='roc_auc', cv=skf) # scoring with CV
        selected_test_X = test_X.loc[:, selected_columns]
        pred = pipeline.predict(selected_test_X)
        auc = roc_auc_score(pred, test_y)
        accuracy = accuracy_score(pred, test_y)
        print("###########################################")
        print(f'train auc: [{np.mean(results)}]')
        print(f'final {pipe_name} auc: [{auc}]')
        print(f'final {pipe_name} accuracy: [{accuracy}]')


if __name__ == '__main__':
  main()
