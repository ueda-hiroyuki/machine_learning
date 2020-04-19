import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from collections import OrderedDict
from python_file.practice.deep_learning.common_layers import *
from sample_data.deep_learning_documents.dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads


class MultiLayerNetExtend:
    def __init__(
        self, 
        input_size, 
        hidden_size_list, 
        output_size, 
        activation = 'relu',
        weight_init_std = 'relu', # 重みの初期値
        weight_decay_lambda = 0, # 重み減衰時の正則化の強さを決定するパラメータ
        dropout = False, # Dropoutは使用するのか
        dropout_ratio = 0.5, # Dropoutさせるニューロンの閾値(数)
        batch_normal = False, # バッチ正規化は使用するのか
    ):
        self.layers = OrderedDict()
        self.params = {}
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_size_num = len(hidden_size_list)
        self.dropout = dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.batch_normal = batch_normal
        self.all_list_size = len([input_size] + hidden_size_list + [output_size])

        # 重みの初期化(学習が効率よく進むような初期値)
        self._init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu} # 活性化関数を適応する層
        for i in range(1, self.hidden_size_num+1):
            # Affineレイヤ
            self.layers[f'Affine{i}'] = Affine(self.params[f"W{i}"], self.params[f"b{i}"])
            # (バッチ正規化レイヤ)
            if self.batch_normal:
                self.params[f'gamma{i}'] = np.ones(hidden_size_list[i-1]) # 前層のニューロン数の要素が1の配列を作成する
                self.params[f'beta{i}'] = np.zeros(hidden_size_list[i-1]) # 前層のニューロン数の要素が0の配列を作成する
                self.layers [f'BatchNorm'] = BatchNormalization(self.params[f'gamma{i}'], self.params[f'beta{i}'])
        
            # アクティベーションレイヤ
            self.layers[f'Activation{i}'] = activation_layer[activation]() # 引数として与えた値に応じて活性化関数を変える

            # (ドロップアウトレイヤ)
            if self.dropout:
                self.layers[f'Dropout{i}'] = Dropout(dropout_ratio)

        # 出力層
        idx = self.hidden_size_num+1 # 入力層+隠れ層
        self.layers[f'Affine{idx}'] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
        self.lastLayer = SoftmaxWithLoss() # 最終層にはソフトマックス関数+交差エントロピー誤差を適応


    def _init_weight(self, weight_init_std):
        # weight_init_stdで'sigmoid','xavier'を指定した場合は「Xavierの初期値」
        # weight_init_stdで'relu','he'を指定した場合は「heの初期値」
        all_list_size = [self.input_size] + self.hidden_size_list + [self.output_size] # 入力層、隠れ層、出力層の数
        for i in range(1, len(all_list_size)):
            if weight_init_std in ('relu', 'he'):
                scale = np.sqrt(2/all_list_size[i-1]) # Heの初期値を設定する場合、標準偏差としてsqrt(2/n)を用いる(nは前層のニューロンの数)
            elif weight_init_std in ('sigmoid', 'xavier'):
                scale = np.sqrt(1/all_list_size[i-1]) # Xavierの初期値を設定する場合、標準偏差としてsqrt(1/n)を用いる(nは前層のニューロンの数)
            self.params[f"W{i}"] = scale * np.random.randn(all_list_size[i-1], all_list_size[i])
            self.params[f"b{i}"] = np.zeros(all_list_size[i])
        
    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNormal" in key:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t, train_flag=False):
        y = self.predict(x, train_flag)
        weight_decay = 0
        # 各層における重みにペナルティを加算する(重みの増加を抑えることで過学習を抑制できる)
        for i in range(1, self.all_list_size):
            W = self.params[f'W{i}']
            weight_decay += 1/2 * self.weight_decay_lambda * np.sum(W**2) 
        # 出力された損失関数にL2正則化項を加算する
        return self.lastLayer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flag=True)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 設定
        grads = {}
        for i in range(1, self.all_list_size):
            # 重みの勾配を求める場合、誤差逆伝播法の結果にλWを加算すればよい(λW：1/2*λW**2の微分値)
            grads[f'W{i}'] = self.layers[f'Affine{i}'].dW + self.weight_decay_lambda * self.params[f'W{i}']
            grads[f'b{i}'] = self.layers[f'Affine{i}'].db
            if self.batch_normal and i != self.all_list_size-1:
                grads[f'gamma{i}'] = self.layers[f'BatchNorm'].dgamma
                grads[f'beta{i}'] = self.layers[f'BatchNorm'].dbeta

        return grads