import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t

logging.basicConfig(level=logging.INFO)


class MultiLayer: # 乗算レイヤ
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, d_out):
        dx = d_out * self.y
        dy = d_out * self.x
        return dx, dy

class AddLayer: # 加算レイヤ
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, d_out):
        dx = d_out
        dy = d_out
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x(numpy.array)の0以下の部分をTrue、0より大きい部分はFalseとする
        out = x.copy()
        out[self.mask] = 0 # maskの情報から、0以下の部分は0、0より大きい部分はx(そのままの値)を返す
        return out 
    
    def backward(self, d_out):
        d_out[self.mask] = 0
        d_x = d_out
        return d_x

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out 
        return out

    def backward(self, d_out):
        out = self.out
        dx = d_out * out(1.0 - out)
        return d_x

class Affine(self):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.d_W = None
        self.d_b = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, d_out):
        d_x = np.dot(d_out, self.W.T)
        self.d_W = np.dot(self.x.T, d_out)
        self.d_b = np.sum(d_out, axis=0)
        return d_x

class SoftmaxWithLoss: # 活性化関数SoftMaxを用いる場合(分類)、損失関数として交差エントロピー誤差を用いる。
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmax関数の出力
        self.t = None # 教師ラベル(one-hot-vector)
        self.common = CommonFunctions()   

    def forward(self, x, t):
        self.t = t
        self.y = softmax_func(x)
        self.loss = self.common.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0] # 教師データの"行"に相当する部分
        d_x = (self.y - self.t) / batch_size # バッチサイズで割ることでデータ1つ当たりの誤差を前層に伝搬することができる。
        return　d_x
        
    
class CommonFunctions:
    def __init__(self):
        pass

    def softmax_func(a):
        c = np.max(a)
        exp_a = np.exp(a - c) # オーバーフロー対策
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a
    
    def sigmoid_func(x):
        y = 1 / (1 + np.exp(-x))
        return y 

    def mean_squared_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
        e = 1/2 * (np.sum((y-k)**2))
        return e

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 教師データ(t)がone-hot-vectorの場合、正解ラベルのインデックスに変換される ⇒ ex) [0,0,1,0,0,0] = [2]
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def main():

if __name__ == "__main__":
    main()