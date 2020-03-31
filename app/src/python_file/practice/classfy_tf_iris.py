import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


labels = {
    'Iris-setosa': [1,0,0],
    'Iris-versicolor': [0,1,0],
    'Iris-virginica': [0,0,1]
}

def softmax(x):
    w = tf.Variable(tf.zeros([4,3]))
    b = tf.Variable(tf.zeros([3]))

    y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y


if __name__ == "__main__":
    iris_data = pd.read_csv("src/sample_data/iris.csv")
    print(iris_data)
    label = iris_data.loc[:, "Name"]
    data = iris_data.drop("Name", axis=1)
    y = list(map(lambda x: labels[x],label))
    


