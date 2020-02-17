import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

iris_data = pd.read_csv("../sample_data/iris.csv")

label = iris_data.loc[:, "Name"]
data = iris_data.drop("Name", axis=1)

# tenforflowにおいて多クラス分類行う場合はワンホットベクトルを定義する ⇒ 0 or 1
one_hot_vec = {
    'Iris-setosa': [1,0,0],
    'Iris-versicolor': [0,1,0],
    'Iris-virginica': [0,0,1]
}

# label(分類を行うアヤメの品種名の配列)からn行3列(3値分類するため)のdataframeを作成
y_num = list(map(lambda v: one_hot_vec[v], label))

# 学習データとテストデータに分割する
x_train, x_test, y_train, y_test = train_test_split(data, y_num, test_size=0.2)

# 入力値、出力値の入れ物(型)を定義
x = tf.placeholder(tf.float32, [None, 4]) # 入力値はSepalLength,SepalWidth,PetalLength,PetalWidthの4次元
_y = tf.placeholder(tf.float32, [None, 3]) # 出力値はIris-setosa,Iris-versicolor,Iris-virginicaの3次元

# 重みとバイアスを定義 ⇒ 重み：各ニューロン間の結びつきの強さ、バイアス：ニューロンの偏り
# y = x*w + b (y:出力値、x:入力値、w:重み、b:バイアス)
weight = tf.Variable(tf.zeros([4,3])) # 要素がすべて0の4行3列配列
bias = tf.Variable(tf.zeros([3])) # 要素がすべて0の1行3列配列

# ソフトマックス回帰 ⇒ 無制限の実数範囲の入力xに対して、0~1までの値であるyに変換して出力する関数(yの合計は1となる)
# softmax関数は多クラス分類に向いている
y = tf.nn.softmax(tf.matmul(x, w) + b) # tf.matmul: 引数として2つの行列を渡すことでそれらの行列の積を返す。




