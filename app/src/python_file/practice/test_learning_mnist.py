import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.dataset import mnist as mn


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_func(z)
        loss = cross_entropy_error(y, t)
        return loss


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train


def mean_squared_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
    e = 1/2 * (np.sum((y-k)**2))
    return e


def cross_entropy_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    e = - np.sum(t * np.log(y + delta)) / batch_size # +deltaはlog0(-inf)とならないような対策
    return e


def numerical_diff(f, x): #数値微分
    h = 1e-4
    e = (f(x+h) - f(x-h)) / 2*h # 前方差分((f(x+h)-f(x))/h)よりも中心差分を使用する
    return e


# 勾配ベクトルの算出
def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
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
    batch_size = 10
    x_train, t_train = load_data()
    train_size = x_train.shape[0]
    
    batch_mask = np.random.choice(train_size, batch_size) 

    batch_x_train = x_train[batch_mask]
    batch_t_train = t_train[batch_mask]

def main():
    simple_net = simpleNet()
    x = np.array([0.6, 0.9])
    p = simple_net.predict(x)
    t = np.array([0,0,1])
    loss = simple_net.loss(x,t)
    print(loss)

if __name__ == "__main__":
    main()