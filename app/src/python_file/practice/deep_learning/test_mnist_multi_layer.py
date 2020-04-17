import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from collections import OrderedDict
from sample_data.deep_learning_documents.common.gradient import numerical_gradient

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


class TwoLayerBackProp:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        weight_init_std=0.01,
    ):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = weight_init_std * np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = weight_init_std * np.zeros(output_size)
        
        # レイヤの生成(入力層⇒隠れ層)
        self.layers = OrderedDict() # 追加した順番を認知可能な辞書型
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"] )
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"] )

        # レイヤの生成(隠れ層⇒出力層)
        self.last_layer = SoftmaxWithLoss()

        # 最終層からの信号
        self.d_out = 1

    # 認識(推論)を行う(引数xは画像データ)
    def predict(self, x):
        for layer in self.layers.values(): # Affine1, Relu1, Affine2の計算を行う。
            x = layer.forward(x) # 前層の入力から算出した出力が次層の入力となる。
        return x 

    # 損失関数の算出    
    def calc_loss(self, x, t): # x:入力データ、t:教師データ
        y = self.predict(x)
        loss_func = self.last_layer.forward(y, t) # 最終層で損失関数の値を算出
        return loss_func
    
    # 認識算出の算出
    def calc_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if y.ndim != 1: # バッチ対応
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

    def calc_numerical_gradient(self, x, t):
        loss_func = lambda W: self.calc_loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_func, self.params["W1"])
        grads["W2"] = numerical_gradient(loss_func, self.params["W2"])
        grads["b1"] = numerical_gradient(loss_func, self.params["b1"])
        grads["b2"] = numerical_gradient(loss_func, self.params["b2"])
        return grads

    # 重みパラメータに対する勾配(誤差関数の傾き)を誤差逆伝播法により算出
    def calc_gradient(self, x, t):
        # forward
        loss_func = self.calc_loss(x, t)
        # backward
        d_out = self.last_layer.backward(self.d_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers: # Affine2 ⇒ Relu1 ⇒ Affine1と誤差を逆伝播する。
            d_out = layer.backward(d_out)

        # 各層における勾配
        grads = {}
        grads["W1"] = self.layers["Affine1"].d_W
        grads["b1"] = self.layers["Affine1"].d_b
        grads["W2"] = self.layers["Affine2"].d_W
        grads["b2"] = self.layers["Affine2"].d_b
        return grads
        

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x(numpy.array)の0以下の部分をTrue、0より大きい部分はFalseとする
        out = x.copy()
        out[self.mask] = 0 # maskの情報から、0以下の部分は0、0より大きい部分はx(そのままの値)を返す
        return out 
    
    def backward(self, d_out):
        d_out[self.mask] = 0 # Trueの部分(0以下の部分)は0で返す
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

# 行列の積を算出するクラス(y=x*w+b)
class Affine:
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
        self.y = self.common.softmax_func(x)
        self.loss = self.common.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0] # 教師データの"行"に相当する部分
        d_x = (self.y - self.t) / batch_size # バッチサイズで割ることでデータ1つ当たりの誤差を前層に伝搬することができる。
        return d_x


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
 

def main():
    two_layer_net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10, weight_init_std=0.01)
    x = np.array([[1,1,0],[2,3,4]])
    two_layer_net.predict(x)

if __name__ == "__main__":
    main()