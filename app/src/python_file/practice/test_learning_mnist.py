import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.dataset import mnist as mn


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
    print(e)
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
        fake_x = x 
        val = fake_x[idx]
        
        fake_x[idx] = val + h
        f1 = f(fake_x) # f(x+h)

        fake_x[idx] = val - h
        f2 = f(fake_x) # f(x-h)

        grad[idx] = (f1-f2) / (2*h)
    return grad


def func_2(x):
    f = np.sum(x**2)
    return f


def exec():
    batch_size = 10
    x_train, t_train = load_data()
    train_size = x_train.shape[0]
    
    batch_mask = np.random.choice(train_size, batch_size) 

    batch_x_train = x_train[batch_mask]
    batch_t_train = t_train[batch_mask]

def main():
    diff = numerical_grad(func_2, np.array([3.0,4.0]))
    print(diff)


if __name__ == "__main__":
    main()