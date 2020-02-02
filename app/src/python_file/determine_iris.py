import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC # svm: サポートベクターマシーン(データ集合を分類する識別関数), SVC: support vector classification
#　サポートベクターマシーンはデータ数が少ない際に用いられることが多い。

iris_data = pd.read_csv("../sample_data/iris.csv", encoding="utf-8") # encoding="utf-8"：csvファイルを文字化けせずに読み込む 

x = iris_data.loc[:,"SepalLength":"PetalWidth"]
y = iris_data.loc[:,"Name"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = SVC().fit(x_train, y_train)
y_pred = clf.predict(x_test)

score = accuracy_score(y_test, y_pred) 

print(f'正解率は {score} です')
