import glob
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # プログレスバー表示

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from python_file.practice.pytorch.section1.transfer_traing import *


def main():
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32  # ミニバッチサイズの設定
    num_epoch = 5
    is_use_pretrained = True

    path_list_for_train = make_datapath_list(phase="train")
    path_list_for_valid = make_datapath_list(phase="valid")

    train_dataset = HymenopteraDataset(
        file_list=path_list_for_train,
        transform=ImageTransform(resize, mean, std),
        phase="train",
    )
    valid_dataset = HymenopteraDataset(
        file_list=path_list_for_valid,
        transform=ImageTransform(resize, mean, std),
        phase="valid",
    )

    train_dataloader = data.DataLoader(
        train_dataset, batch_size, shuffle=True
    )  # 学習用Dataloader: データセットからデータをバッチサイズに固めて返すモジュール(Tensor型：勾配計算時必須)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size, shuffle=False
    )  # 検証用Dataloader

    data_loaders_dict = {"train": train_dataloader, "valid": valid_dataloader}
    net = models.vgg16(
        pretrained=is_use_pretrained
    )  # vgg16は「特徴抽出部(features)」と「クラス分類部(classifier)」の2部に分かれている。

    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.train()

    updated_params1 = []
    updated_params2 = []
    updated_params3 = []
    learning_params_name1 = ["features"]  # vgg16のfeatures部分
    learning_params_name2 = [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    ]  # vgg16.classifierの前半2つ
    learning_params_name3 = [
        "classifier.6.weight",
        "classifier.6.bias",
    ]  # vgg16.classifierの最終層

    for name, param in net.named_parameters():
        if learning_params_name1[0] in name:
            param.requires_grad = True
            updated_params1.append(param)

        if name in learning_params_name2:
            param.requires_grad = True
            updated_params2.append(param)

        if name in learning_params_name3:
            param.requires_grad = True
            updated_params3.append(param)
        else:
            param.requires_grad = False

    print(updated_params1, updated_params2, updated_params3)


if __name__ == "__main__":
    main()