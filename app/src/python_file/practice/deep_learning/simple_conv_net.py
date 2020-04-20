import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from collections import OrderedDict
from python_file.practice.deep_learning.common_func import *
from python_file.practice.deep_learning.common_layers import *

logging.basicConfig(level=logging.INFO)

# (conv1 ⇒ Relu1 ⇒ Pool1) ⇒ (Affine1 ⇒ Relu2) ⇒ (Affine2 ⇒ SoftMax) のニューラルネットワーク
class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1,28,28), # 入力画像1枚の次元(チャンネル数:1、幅:28、高さ:28)
        conv_param={
            'filter_num': 30, # フィルターの枚数
            'filter_size': 5, # フィルターの幅
            'padding': 0,
            'stride': 1
        },
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01 # 重みの標準偏差
    ):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_padding = conv_param['padding']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_padding) / filter_stride + 1
        # プーリング層では特徴マップに対し2×2ずつプーリングを行うため、出力サイズは高さ、幅共に2で割る(画像枚数(filterの枚数)はそのまま)
        # 出力サイズはreshape前の縦サイズ(reshape後は入力サイズの高さ、幅が半分(面積は1/4)となる)
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        print(filter_num, filter_size, input_size, conv_output_size, pool_output_size)

        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param['stride'],
            conv_param['padding']
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(
            self.params["W2"],
            self.params["b2"],
        )
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(
            self.params["W3"],
            self.params["b3"],
        )
        self.last_layer = SoftmaxWithLoss()
    
    
    def predict(self, x):
        for name, layer in self.layers.items():
            logging.info(f"layer_name : {name}")
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
    

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
