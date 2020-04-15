import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from python_file.practice.deep_learning.test_mnist_multi_layer import TwoLayerNet
from sample_data.deep_learning_documents.dataset import mnist as mn


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train, x_test, t_test


def main():
    two_layer_net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01)
    x_train, t_train, x_test, t_test = load_data()

    x_train_batch = x_train[:3]
    t_train_batch = t_train[:3]

    # grad_numerical = two_layer_net.calc_numerical_gradient(x_train_batch, t_train_batch)
    grad_backprop = two_layer_net.calc_gradient(x_train_batch, t_train_batch)

    #print(grad_numerical, grad_backprop)

if __name__ == "__main__":
    main()