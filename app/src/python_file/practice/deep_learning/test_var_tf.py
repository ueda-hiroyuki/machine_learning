import tensorflow as tf

# 変数を定義
v = tf.Variable(0, name="v") # 引数：(初期値, 名前）

# 定数を定義
a = tf.constant(10, name="10")
b = tf.constant(20, name= "20")

# 演算を定義
mul = tf.multiply(a, b, name="mul")

# 変数に演算結果をアサインする。
result_mul = tf.assign(v, mul) # 変数vにmul(商)を代入する

# 演算を実行する
sess = tf.Session()
sess.run(result_mul)

# 演算のグラフを表示
tf.summary.FileWriter("../sample_data/logs", sess.graph)

res = sess.run(v)
print(res)

