import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.dataset import mnist as mn
from PIL import Image


def init_network():
    network = joblib.load("src/sample_data/mnist/params_backprop.pkl")
    return network


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def sigmoid_func(x):
    y = 1 / (1 + np.exp(-x))
    return y 


def softmax_func(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_func(a3)
    return y

def predict_two_later(network, x):
    W1, W2 = network["W1"], network["W2"]
    b1, b2 = network["b1"], network["b2"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax_func(a2)
    return y
    

def main():
    batch_size = 100
    network = init_network()
    x_test, t_test = load_data()
    accuracy_cnt = 0
    for idx in range(0, len(x_test), batch_size):
        pred = predict_two_later(network, x_test[idx: idx + batch_size])
        p = np.argmax(pred, axis=1)
        match = np.sum(p == t_test[idx: idx + batch_size])
        accuracy_cnt += match
    print(f"Score is {accuracy_cnt/len(t_test)}")


if __name__ == "__main__":
    main()
