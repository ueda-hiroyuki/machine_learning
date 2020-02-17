import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import joblib

digits = datasets.load_digits()

x = digits.images.reshape((-1, 64))
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = svm.LinearSVC().fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

joblib.dump(clf, "../sample_data/digits.pkl") # 学習器の保存
