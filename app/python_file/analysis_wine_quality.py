import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("../sample_data/winequality-white.csv", sep=";", encoding="utf-8")

x = df.drop("quality", axis=1) # データ
y = df["quality"] # ラベル

new_list = []
for v in list(y):
    if v <= 4:
        new_list = new_list + [0]
    elif v <= 7:
        new_list = new_list + [1]
    else:
        new_list = new_list + [2]    

y = new_list    

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = rfc().fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
print("正答率 ＝", accuracy_score(y_test, y_pred))