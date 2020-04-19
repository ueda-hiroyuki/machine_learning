import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t


class CommonFunctions:
    def identity_function(self, x):
        return x


    def step_function(self, x):
        return np.array(x > 0, dtype=np.int)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    


    def sigmoid_grad(self, x):
        return (1.0 - sigmoid(x)) * sigmoid(x)
        

    def relu(self, x):
        return np.maximum(0, x)


    def relu_grad(self, x):
        grad = np.zeros_like(x)
        grad[x>=0] = 1
        return grad
        

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


    def sum_squared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)


    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
                
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    # 畳み込み層における出力
    def conv_output_size(self, input_size, filter_size, stride=1, pad=0):
        return (input_size + 2*pad - filter_size) / stride + 1

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]