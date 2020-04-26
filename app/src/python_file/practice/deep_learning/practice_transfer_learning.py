import os
import json
import logging
import joblib
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict

"""

=== 転移学習とファインチューニングの勉強 ===
● 転移学習とファインチューニングの違い
⇒ 転移学習：事前学習されたニューラルネットワークの重みパラメータは固定し、新たに追加したレイヤの重みのみを再学習させる手法
⇒ ファインチューニング：事前学習されたモデルの重みパラメータを新規データをもとに、モデル全体の重みを再学習させる手法

"""

PATH = 'src/sample_data/images/screen.jpg'

def run_detect(image_path):   
    # vgg16のモデルを事前学習させておく
    network = models.vgg16(pretrained=True) 
    network.eval()
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # jsonファイル内のラベリングデータを取得
    class_index = json.load(open('src/sample_data/Image-Classifier/data/imagenet_class_index.json', 'r'))
    image = Image.open(image_path)
    loader = transforms.Compose([
        transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
        transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
        transforms.ToTensor(),  # Torchテンソルに変換
        transforms.Normalize(mean, std)  # 色情報の標準化
    ])
    image = loader(image).unsqueeze(0) # バッチサイズの次元を追加
    out = network(image)  # torch.Size([1, 1000])
    predicted = torch.argmax(out, dim=1)
    predicted = predicted.numpy()[0]
    detection = class_index[f'{predicted}'][1]
    print(f'{detection}の写真です!')


def imshow(input, title):
    print(input.shape)
    input = input.numpy().transpose((1, 2, 0))
    # 正規化(transforms.Normalize)の逆処理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.savefig('src/sample_data/images/samples/hymenoptera.jpg')


def main():
    batch_size = 10
    data_dir = "src/sample_data/images/hymenoptera_data"
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # 指定したアスペクト比でリサイズする
            transforms.RandomHorizontalFlip(), # 画像を左右反転する
            transforms.ToTensor(),
            # データセット全体のデータで正規化(標準化)すると効率よく重みを更新でき、スムーズに学習を進めることができる。
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # 指定したサイズで中心部分をトリミング
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms["train"])
    val_data = datasets.ImageFolder(root=f"{data_dir}/val", transform=data_transforms["val"])
    train_batch = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_batch = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # inputs, classes = next(iter(train_batch))
    # out = torchvision.utils.make_grid(inputs, nrow=5)
    # print(inputs)
    # imshow(out, title=[c for c in classes.numpy()])



if __name__ == "__main__":
    main()