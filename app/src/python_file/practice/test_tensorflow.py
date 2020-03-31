import tensorflow as tf

w = tf.Variable(tf.ones(shape=(2,2)), name="w")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def culc1(x): 
    print(w, x, b) 
    result = w * x + b
    return result

@tf.function
def culc2():
    a = 100 # 定数の定義
    b = 4
    c = 3
    add = a + b # 演算の定義
    mul = add * c
    return mul

if __name__ == "__main__":
    result = culc2()
    print(result)