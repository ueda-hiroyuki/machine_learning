import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

wine_data = pd.read_csv("../sample_data/winequality-white.csv", sep=";", encoding="utf-8")

x = wine_data.loc[:,"fixed acidity":"alcohol"]
y = wine_data.loc[:,"quality"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = RandomForestClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)

score = accuracy_score(y_test, y_pred)
print(f"正解率は {score} です")
print(classification_report(y_test, y_pred)) # clasiffication_reportは各クラスにおける正解率、適合率、再現率を一度に算出してくれる

test = y_test.values.tolist()
print(type(test), type(y_pred))
plt.plot(test[:100], c="red")
plt.plot(y_pred[:100], c="blue")
plt.show()
