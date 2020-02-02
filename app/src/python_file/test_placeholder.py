import tensorflow as tf

a = tf.placeholder(tf.int32, [5]) # 変数aの入れ物(形のようなもの)を定義 ⇒ 1行5列配列にint32型の数字が入る

two = tf.constant(2)
mul_two = a * two

sess = tf.Session()

# 演算を実行する際の引数として、演算自体とfeed_dictでplaceholderで指定した形の数値を与える。
res1 = sess.run(mul_two, feed_dict={a: [1,2,3,4,5]})
print(res1)

res2 = sess.run(mul_two, feed_dict={a: [10,20,30,40,50]})
print(res2)


