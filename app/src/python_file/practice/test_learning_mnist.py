import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.dataset import mnist as mn


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_func(z)
        loss = cross_entropy_error(y, t)
        return loss

class TwoLayerNet:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        weight_init_std = 0.01,
    ):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid_func(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid_func(a2)
        y = softmax_func(z2)
        return y

    # x:入力データ, t:教師データ
    def calc_loss(self, x, t):
        y = self.predict(x)
        loss_func = cross_entropy_error(y, t)
        return loss_func
        
    def calc_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def calc_gradient(self, x, t):
        loss_func = lambda W: self.calc_loss(x, t)
        print(f"loss_func = {loss_func}")
        grads = {}
        grads["W1"] = numerical_grad(loss_func, self.params["W1"])
        grads["W2"] = numerical_grad(loss_func, self.params["W2"])
        grads["b1"] = numerical_grad(loss_func, self.params["b1"])
        grads["b2"] = numerical_grad(loss_func, self.params["b2"])
        return grads


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train, x_test, t_test


def mean_squared_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
    e = 1/2 * (np.sum((y-k)**2))
    return e


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(f, x): #数値微分
    h = 1e-4
    e = (f(x+h) - f(x-h)) / 2*h # 前方差分((f(x+h)-f(x))/h)よりも中心差分を使用する
    return e


# 勾配ベクトルの算出
def numerical_grad(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext() 
    return grad


# 勾配降下法
def gradient_descent(func, init_x, lr=0.01, step_num=100): # 勾配降下法
    x = init_x
    for n in range(step_num):
        grad = numerical_grad(func, x) # 勾配の算出
        print(grad)
        x = x - lr * grad
    return x


def func_2(x):
    f = np.sum(x**2)
    return f


def sigmoid_func(x):
    y = 1 / (1 + np.exp(-x))
    return y 


def softmax_func(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def exec():
    iter_num = 10000
    batch_size = 100
    learning_rate = 0.1
    x_train, t_train, x_test, t_test = load_data()
    train_size = x_train.shape[0]
    print(x_train.shape)

    two_layer_net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size) # 全テストデータからランダムにbatch_size分だけ選択(return list of index)
        x_train_batch = x_train[batch_mask]
        t_train_batch = t_train[batch_mask]


def main():
    two_layer_net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    x = np.random.rand(100,784) 
    t = np.random.rand(100, 10)

    grads = two_layer_net.calc_gradient(x, t)
    print(grads)


if __name__ == "__main__":
    exec()