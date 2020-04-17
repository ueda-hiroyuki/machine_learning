import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from python_file.practice.deep_learning.test_mnist_multi_layer import TwoLayerBackProp
from sample_data.deep_learning_documents.dataset import mnist as mn


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train, x_test, t_test


def main():
    two_layer_net = TwoLayerBackProp(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)
    x_train, t_train, x_test, t_test = load_data()

    x_train_batch = x_train[:3]
    t_train_batch = t_train[:3]

    # 数値微分で求めた勾配
    grad_numerical = two_layer_net.calc_numerical_gradient(x_train_batch, t_train_batch)
    # バックプロパゲーションで求めた勾配
    grad_backprop = two_layer_net.calc_gradient(x_train_batch, t_train_batch)

    for key in grad_numerical.keys(): # key = W1,W2,b1,b2
        diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
        print(f"diff_{key} : {diff}")



if __name__ == "__main__":
    main()