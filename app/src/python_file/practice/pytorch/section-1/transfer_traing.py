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


DATA_DIR = "src/sample_data/pytorch_advanced/1_image_classification/data"
HYMENOPTERA_DATA_DIR = (
    "src/sample_data/pytorch_advanced/1_image_classification/data/hymenoptera_data"
)


def make_datapath_list(phase="train"):
    """
    データのパスを格納したリストを作成する。
    ・input
        phase：'train' or 'valid'
        ⇒ 学習用データ、または検証用データかを指定する。
    ・output
        path_list：list
        ⇒ データへのパスを格納したリスト
    """
    target_path = f"{HYMENOPTERA_DATA_DIR}/{phase}/**/*.jpg"
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list


class ImageTransform:
    """
    画像の前処理クラス(学習時と検証時で違う挙動とする)
    ⇒ 画像をリサイズし、色の標準化を行う。
    ⇒ 学習時にはData Augumentationを行う。

    ・input
        resize：リサイズ後の画像の大きさ
        mean：各チャネルの色の平均
        std：各チャネルの色の標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        resize, scale=(0.5, 1)
                    ),  # 指定したサイズに調整しData Augumentationする
                    transforms.RandomHorizontalFlip(),  # 50%の確率で画像を左右反転させてData Augumentationする
                    transforms.ToTensor(),  # テンソル型に変換
                    transforms.Normalize(mean, std),  # 色の標準化
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        """
        input: phase(train or valid)

        """
        return self.data_transforms[phase](img)


class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラスpytorchの「Datasetクラス」を継承している。
    ⇒ 画像データとそれに対応するラベルを1組返すモジュール
    ⇒ 上記画像データはtransformsにより前処理済みのものである(従って、Datasetを作成する際にはtransformsを引数として渡す必要がある)

    〇Datasetクラスを自作する条件
    ⓵pytorchのDatasetクラスを継承する
    ⓶__len__を実装する ⇒ __len__は、len(obj)で実行されたときにコールされる関数。
    ⓷__getitem__を実装する ⇒ __getitem__は、obj[i]のようにインデックスで指定されたときにコールされる関数。

    ・input
        path_list：画像のpathを格納したリスト
        transform：前処理クラスのインスタンス
        phase：'train'または'valid'
    """

    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)  # 画像の枚数を返す

    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとラベルを取得する
        """

        img_path = self.file_list[index]
        img = Image.open(img_path)

        transformed_img = self.transform(
            img, "train"
        )  # 画像の前処理の実施(call関数実行) ⇒ Tensor([3,224,224])

        # 画像のラベル(アリorハチ)をpathから抜き出す
        if "ants" in img_path:
            label = "ants"
        elif "bees" in img_path:
            label = "bees"
        else:
            label = ""

        # ラベルを数値に変換する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return transformed_img, label


def main():
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32  # ミニバッチサイズの設定
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
    )  # 検証用用Dataloader

    data_loaders_dict = {"train": train_dataloader, "valid": valid_dataloader}
    net = models.vgg16(
        pretrained=is_use_pretrained
    )  # vgg16は「特徴抽出部(features)」と「クラス分類部(classifier)」の2部に分かれている。

    net.classifier[6] = nn.Linear(
        in_features=4096, out_features=2
    )  # vgg16の出力層を置換している(出力は「アリ」or「ハチ」の2種類)
    net.train()  # netを学習モードに変更

    criterion = nn.CrossEntropyLoss()  # 分類問題の損失関数は基本的に「クロスエントロピー誤差関数」を用いる。

    # for i, (name, param) in enumerate(net.named_parameters()): # name: どこの層のパラメータなのか　param： 各層における学習済みのweightまたはbias
    #     print(f"###################{i}##################")
    #     print(name, param)
    # print(net)


if __name__ == "__main__":
    main()