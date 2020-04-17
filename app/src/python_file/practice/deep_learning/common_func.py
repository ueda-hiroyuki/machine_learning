import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t


class CommonFunctions:
    def identity_function(self, x):
        return x


    def step_function(self, x):
        return np.array(x > 0, dtype=np.int)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    


    def sigmoid_grad(self, x):
        return (1.0 - sigmoid(x)) * sigmoid(x)
        

    def relu(self, x):
        return np.maximum(0, x)


    def relu_grad(self, x):
        grad = np.zeros_like(x)
        grad[x>=0] = 1
        return grad
        

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


    def sum_squared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)


    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

