import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp == 1:
        return 1
    else:
        return 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp > 0:
        return 1
    else:
        return 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    tmp = np.sum(x*w) 
    if tmp == 1:
        return 0
    else:
        return 1

def XOR(x1, x2):
    res1 = NAND(x1,x2)
    res2 = OR(x1,x2)
    res = AND(res1, res2)
    return res


def step_func(x):
    print(x)
    y = x > 0
    print(y)
    return y.astype(np.int)


def sigmoid_func(x):
    y = 1 / (1 + np.exp(-x))
    return y 

def relu_func(x):
    return np.maximum(0, x) # 大きい値を返す

def calc_dot(x, y):
    return np.dot(x,y)

def calc_out(x, w, b):
    a = np.dot(x, w) + b 
    return a

def softmax_func(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def three_layer_perceptron():
    x = np.array([1.0, 0.5])
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])

    a1 = calc_out(x,w1,b1)
    z1 = sigmoid_func(a1)

    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    a2 = calc_out(z1, w2, b2)
    z2 = sigmoid_func(a2) 

    w3 = np.array([[0.1, 0.3],[0.2,0.4]])
    b3 = np.array([0.1,0.2])

    a3 = calc_out(z2, w3, b3)  
    # y = a3 # 恒等関数
    y = softmax_func(a3)
    print(y)



def main() -> None:
    a = np.array([2.0, 0.5, 3.5])
    y = softmax_func(a)
    print(y)
    print(np.sum(y)) # np.sum(y)は1となる 
    


if __name__ == "__main__":
    main()