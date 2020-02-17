import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def kesson_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns


def create_dataset():
    train_data = pd.read_csv("../Kaggle/kaggle_dataset/titanic/train.csv")
    test_data = pd.read_csv("../Kaggle/kaggle_dataset/titanic/test.csv")

    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    train_data = train_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}})
    test_data = test_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}})
    test_data.Fare[152] = test_data.Fare.median()

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = create_dataset()
    train_x = train_data.loc[:,["Pclass","Sex", "Age","SibSp","Parch","Fare", "Embarked"]]
    train_y = train_data['Survived']
    test_x = test_data.loc[:,["Pclass","Sex", "Age","SibSp","Parch","Fare", "Embarked"]]
    
    model = RandomForestClassifier().fit(train_x, train_y)
    y_pred = model.predict(test_x)

    
    # PassengerIdを取得
    PassengerId = np.array(test_data["PassengerId"]).astype(int)
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("../Kaggle/kaggle_dataset/titanic/titanic_pred.csv", index_label = ["PassengerId"])
    
        
        
