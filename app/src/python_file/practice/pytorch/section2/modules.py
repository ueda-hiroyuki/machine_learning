import torch
import cv2
import torchvision
import torch.nn as nn
import numpy as numpy
import pandas as pd


cfg_vgg = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "MC",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
]  # 畳み込み層や、プーリング層で使用する各層のチャネル数

cfg_extra = [256, 512, 128, 256, 128, 256, 128, 256]


def generate_vgg():
    # 34層にもわたるvggモジュールを自作する。
    layers = []
    in_channels = 3  # 入力チャネル数
    for v in cfg_vgg:
        if v == "M":  # "M"はMaxPooling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif v == "MC":  # "MC"はMaxPoolingで"ceil_mode"がTrue(tensorサイズを求める際に少数点以下切り上げ)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:  # 通常の畳み込み層
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])  # 畳み込み層とrelu関数をセットで入れる
            in_channels = v  # 出力チャネル数が次層の入力チャネル数

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(
        512, 1024, kernel_size=3, padding=6, dilation=6
    )  # delated_convolution
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])
    return nn.ModuleList(
        layers
    )  # ModuleList: 必要な層のリストを渡すことでiterationを作成する(forward pipeline)


def generate_extra():
    layers = []
    in_channels = 1024  # extraモジュールには1024個のチャネルを入力
    layers.append(nn.Conv2d(in_channels, cfg_extra[0], kernel_size=(1)))
    layers.append(
        nn.Conv2d(cfg_extra[0], cfg_extra[1], kernel_size=(3), stride=2, padding=1)
    )
    layers.append(nn.Conv2d(cfg_extra[1], cfg_extra[2], kernel_size=(1)))
    layers.append(
        nn.Conv2d(cfg_extra[2], cfg_extra[3], kernel_size=(3), stride=2, padding=1)
    )
    layers.append(nn.Conv2d(cfg_extra[3], cfg_extra[4], kernel_size=(1)))
    layers.append(nn.Conv2d(cfg_extra[4], cfg_extra[5], kernel_size=(3)))
    layers.append(nn.Conv2d(cfg_extra[5], cfg_extra[6], kernel_size=(1)))
    layers.append(nn.Conv2d(cfg_extra[6], cfg_extra[7], kernel_size=(3)))
    return nn.ModuleList(layers)


def generate_loc_and_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    # loc (location): デフォルトボックスを どのように変形させるか(= offset情報)を出力(4種類)
    # conf (confidence): デフォルトボックスに対する各クラスの信頼度(20+1)を出力する

    loc_layers = []  # 物体位置特定用CNN
    conf_layers = []  # クラス分類用CNN

    # source1に対する畳み込み層
    loc_layers.append(
        nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)
    )  # 出力チャネル数はdbox数*4(オフセット)
    conf_layers.append(
        nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)
    )  # 出力チャネル数はdbox数*21(20クラス+背景)

    # source2に対する畳み込み層
    loc_layers.append(nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1))
    conf_layers.append(
        nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)
    )

    # source3に対する畳み込み層
    loc_layers.append(nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1))
    conf_layers.append(
        nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)
    )

    # source4に対する畳み込み層
    loc_layers.append(nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1))
    conf_layers.append(
        nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)
    )

    # source5に対する畳み込み層
    loc_layers.append(nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1))
    conf_layers.append(
        nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)
    )

    # source6に対する畳み込み層
    loc_layers.append(nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1))
    conf_layers.append(
        nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)
    )

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)