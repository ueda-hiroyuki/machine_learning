import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t


class CommonFunctions:
    def __init__(self):
        pass

    def softmax_func(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c) # オーバーフロー対策
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a
    
    def sigmoid_func(self, x):
        y = 1 / (1 + np.exp(-x))
        return y 

    def mean_squared_error(self, y: t.Sequence, t: t.Sequence) -> t.Sequence:
        e = 1/2 * (np.sum((y-k)**2))
        return e

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 教師データ(t)がone-hot-vectorの場合、正解ラベルのインデックスに変換される ⇒ ex) [0,0,1,0,0,0] = [2]
        if t.size == y.size:
            t = np.argmax(t, axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
 