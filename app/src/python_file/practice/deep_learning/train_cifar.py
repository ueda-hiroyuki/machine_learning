import os
import logging
import joblib
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

IMAGE_LABELS = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
])

class LeNet(nn.Module): # LeNetは畳み込み層、プーリング層が各3層ずつの構造である
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel=3, out_channel=16, kernel_size=3, padding=1)
            nn.ReLU(in_place=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channel=16, out_channel=32, kernel_size=3, padding=1)
            nn.ReLU(in_place=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32)
            nn.Conv2d(in_channel=32, out_channel=64, kernel_size=3, padding=1)
            nn.ReLU(in_place=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )
        self.full_conn = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=500), # 全結合層⇒線形変換:y=x*w+b(in_featuresは直前の出力ユニット(ニューロン)数, out_featuresは出力のユニット(ニューロン)数)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=500, out_features=num_classes) # out_features:最終出力数(分類クラス数)
        )


def show_cifer(data, classes, path):
    H = 10
    W = 10
    fig = plt.figure(figsize=(H, W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.4, wspace=0.4)
    for i, (images, labels) in enumerate(data, 0):
        for k in range(0, images.size()[0]):
            # numpyに変換後、[3, 32, 32] -> [32, 32, 3] に変換
            numpy_array = images[k].numpy().transpose((1, 2, 0))
            plt.subplot(H, W, k+1)
            plt.imshow(numpy_array)
            plt.title("{}".format(classes[labels[k]]), fontsize=12, color = "green")
            plt.axis('off')
        break
    plt.savefig(path)


def train_cifar_by_cnn():
    path = "src/sample_data/cifar"
    batch_size = 100
    epoch = 5
    num_classes = 10

    # torchvisionのtransformsには画像変換系の関数が入っている。
    transform = transforms.Compose([ # Compose関数に変換式を与える
        transforms.Resize((32,32)),
        transforms.ToTensor(), # PIL画像またはnumpy.ndarrayをTensor型に変換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizeは正規化(カラー画像は3つのチャネルを持ち、第一引数：平均、第二引数：標準偏差を与える)
    ])
    train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)
    
    # show_cifer(test_batch, IMAGE_LABELS, f'{path}/cifar_img2.png')
    network = LeNet()


if __name__ == "__main__":
    train_cifar_by_cnn()