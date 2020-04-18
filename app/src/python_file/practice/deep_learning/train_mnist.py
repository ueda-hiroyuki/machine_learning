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


class TwoLayerNetExtend:
    def __init__(
        self, 
        input_size, 
        hidden_size_list, 
        output_size, 
        weight_init_std = 0.01,
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

        # 重みの初期化(学習が効率よく進むような初期値)
        self._init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu} # 活性化関数を適応する層
        for i in range(1, self.hidden_size_num+1):
            self.layers[f'Affine{i}'] = Affine(self.params[f"W{i}"], self.params[f"b{i}"])
            if self.batch_normal:
                ...

    def _init_weight(self, weight_init_std):
        # weight_init_stdで'sigmoid','xavier'を指定した場合は「Xavierの初期値」
        # weight_init_stdで'relu','he'を指定した場合は「heの初期値」
        all_list_size = [self.input_size] + self.hidden_size_list + [self.output_size] # 入力層、隠れ層、出力層の数
        for i in range(1, len(all_list_size)):
            if weight_init_std in ('relu', 'he'):
                scale = np.sqrt(2/all_list_size[i-1]) # Heの初期値を設定する場合、標準偏差としてsqrt(2/n)を用いる(nは前層のニューロンの数)
            elif weight_init_std in ('sigmoid', 'xavier'):
                scale = np.sqrt(1/all_list_size[i-1]) # Xavierの初期値を設定する場合、標準偏差としてsqrt(1/n)を用いる(nは前層のニューロンの数)
            self.params[f"W{i}"] = scale * np.random.randn(all_size_list[i-1], all_size_list[i])
            self.params[f"b{i}"] = np.zeros_like(all_size_list[i])
        
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


def train_mnist():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNetExtend(input_size=784, hidden_size=50, output_size=10)
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)
        
        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
    joblib.dump(network.params, "src/sample_data/mnist/params_backprop.pkl")

if __name__ == "__main__":
    train_mnist()