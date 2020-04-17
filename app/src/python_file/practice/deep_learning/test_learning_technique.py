import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.ch05.two_layer_net import TwoLayerNet
from python_file.practice.deep_learning.common_func import CommonFunctions

# 確率的勾配降下法
class SGD:
    def __init__(self, lr=0.01):
        self. lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# モメンタム
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # 初期化として、paramsのvalueと同型のゼロ配列を作成する。
        if self.v is None:
            self.v = {}
            for key, value in params.items:
                self.v[key] = np.zeros_like(value)
        for key in params.keys:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            self.params[key] += self.v[key]

# AdaGrad(学習が進むに連れ、学習率を減衰させていく手法) ⇒ Adaptive Gradientの略
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h =None

    def update(self, params, grads):
        # 初期化として、paramsのvalueと同型のゼロ配列を作成する。
        if self.h is None:
            self.h = {}
            for key, value in params.items:
                self.h[key] = np.zeros_like(value)
        for key in params.keys:
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h) + 1e-7) # +1e-7はゼロ除算対策
      


def main():    
    x = np.random.randn(1000,100)
    node_num = 100
    hidden_layer_num = 5
    activations = {}
    comnon = CommonFunctions()

    for i in range(hidden_layer_num):
        if i != 0:
            x = activations[i-1]
        # アクティベーション分布(重みの標準偏差の値によって、分布が偏ることがある⇒「表現力の制限問題(分布に偏りがないほうが良い)」)
        # w = np.random.randn(node_num, node_num) * 1 / np.sqrt(node_num) # 活性化関数にsigmoid, tanh関数(S字曲線)を用いるときは「Xavierの初期値」
        w = np.random.randn(node_num, node_num) * (np.sqrt(2) / np.sqrt(node_num)) # 活性化関数にRelu関数を用いるときは「Heの初期値」
        a = np.dot(x, w)
        z = comnon.sigmoid_func(a)
        activations[i] = z 

    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(f"{i}_layer")
        plt.tick_params(
            bottom=False,
            left=False,
            right=False,
            top=False
        )
        plt.hist(a.flatten(), 30, range=(0,1))
    plt.savefig("activation_hist")



if __name__ == "__main__":
    main()