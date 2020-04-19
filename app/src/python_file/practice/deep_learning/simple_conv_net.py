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

# 入力層:1、隠れ層:1、出力層:1の2層ニューラルネットワーク
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
        conv_output_size = (input_size + 2*filter_padding - filter_size) / filter_stride + 1
        pool_output_size = int((input_size - filter_size) / filter_stride + 1)

        print(filter_num, filter_size, input_size, conv_output_size, pool_output_size)



        # # 重みの初期化
        # self.params = {}
        # self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # self.params["b1"] = np.zeros(filter_num)
        # self.params["W2"] = weight_init_std * np.random.randn(filter_num, )
