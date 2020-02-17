import tensorflow as tf

a = tf.placeholder(tf.int32, [None, 2]) # placeholderとしてn行2列の入れ物を定義 ⇒ Noneは任意の数値が入る

two = tf.constant(2)
mul_two = a * two

l = [[1,2],[2,3],[3,4],[4,5]]

sess = tf.Session()
res = sess.run(mul_two, feed_dict={a: l})

print(res)