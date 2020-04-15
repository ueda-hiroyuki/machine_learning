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
    def __iinit__(self):
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


def main():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3 
    tax = 1.1
    d_price = 1
    
    multi_apple_layer = MultiLayer()
    multi_orange_layer = MultiLayer()
    add_price_layer = AddLayer()
    multi_tax_layer = MultiLayer()

    # forward
    apple_price = multi_apple_layer.forward(apple, apple_num)
    orange_price = multi_orange_layer.forward(orange, orange_num)
    price = add_price_layer.forward(apple_price, orange_price)
    total_price = multi_tax_layer.forward(price, tax)

    #backward
    d_price, d_tax = multi_tax_layer.backward(d_price)
    d_apple_price, d_orange_price = add_price_layer.backward(d_price)
    d_apple, d_apple_num = multi_apple_layer.backward(d_apple_price)
    d_orange, d_orange_num = multi_orange_layer.backward(d_orange_price)

    print(d_apple, d_apple_num,d_orange, d_orange_num, d_tax)    




if __name__ == "__main__":
    main()