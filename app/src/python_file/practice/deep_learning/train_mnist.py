import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from collections import OrderedDict
from python_file.practice.deep_learning.common_layers import *
from python_file.practice.deep_learning.multi_layer_net import *
from sample_data.deep_learning_documents.dataset.mnist import load_mnist


def train_mnist_two_layer():
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


def train_mnist_extend():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 過学習を再現するために、学習データを削減
    x_train = x_train[:300]
    t_train = t_train[:300]

    input_size = 784
    hidden_size_list = [100, 100, 100, 100, 100, 100]
    output_size = 10

    network = MultiLayerNetExtend(
        input_size=input_size,
        hidden_size_list=hidden_size_list,
        output_size=output_size,
        dropout=True,
        dropout_ratio=0.2,
        batch_normal=True
    )
    print(network.layers)

if __name__ == "__main__":
    train_mnist_extend()