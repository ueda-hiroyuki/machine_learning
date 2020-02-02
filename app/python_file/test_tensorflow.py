import tensorflow as tf


def say_hello():
    hello = tf.constant('hello') # 定数の定義
    res = sess.run(hello)
    print(res) 

def culc():
    a = tf.constant(100, name='100') # 定数の定義
    b = tf.constant(50, name='50')
    c = tf.constant(3, name='3')
    add = tf.add(a,b, name="add") # 演算の定義
    mul = tf.multiply(add, c, name="mul")
    res = sess.run(mul) # 定義した演算を呼び出して評価する
    print(res)
    tf.summary.FileWriter("../sample_data/logs", sess.graph)

if __name__ == "__main__":
    sess = tf.Session() # セッションの開始
    culc()