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


def train_model(net, data_loaders_dict, criterion, optimizer, num_epoch):
    # epochのループ
    for epoch in range(num_epoch):
        print("#################################")
        print(f"start {epoch+1}/{num_epoch} epoch !!")
        print("#################################")

        # 1 epoch毎に学習と検証を繰り返す。
        for phase in ["train", "valid"]:

            if phase == "train":
                net.train()  # モデルを学習モードにする
            else:
                net.eval()  # モデルを検証モードにする
            epoch_loss = 0.0  # モデルの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確認するため、epochが0の時の学習は省略
            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(
                data_loaders_dict[phase]
            ):  # phaseをキーにしてそれぞれのDataloaderを取得
                optimizer.zero_grad()  # optimizerを初期化

                # 順伝播計算
                with torch.set_grad_enabled(
                    phase == "train"
                ):  # ()内がTrueの時、つまり学習時には順伝播計算を行う。
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 出力と正解ラベルを比較し、lossを算出する。
                    _, preds = torch.max(outputs, 1)  # labelを出力(2次元配列の行方向の最大値indexが返る)
                    if phase == "train":
                        loss.backward()  # 学習時には誤差逆伝播を行う
                        optimizer.step()  # パラメータの更新(勾配計算後に呼び出せる)

                    # イテレーション結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新
                    print(f"loss: {loss.item()}")
                    print(f"size: {inputs.size()}")
                    print(f"shape: {inputs.shape}")

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(data_loaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(data_loaders_dict[phase].dataset)

            print("#################################")
            print(f"{phase} Loss:{epoch_loss}, Acc:{epoch_acc}")
            print("#################################")


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

    net.classifier[6] = nn.Linear(
        in_features=4096, out_features=2
    )  # vgg16の出力層を置換している(出力は「アリ」or「ハチ」の2種類)
    net.train()  # netを学習モードに変更

    updated_params = []  # 学習させるパラメータを格納する
    learning_params_name = [
        "classifier.6.weight",
        "classifier.6.bias",
    ]  # 再学習させるのは最終層の重みとバイアスのみ(転移学習)

    for name, param in net.named_parameters():
        if name in learning_params_name:
            param.requires_grad = True  # Trueの場合は勾配が再計算される。
            updated_params.append(param)
        else:
            param.requires_grad = False  # Falseの場合は勾配は固定される

    criterion = nn.CrossEntropyLoss()  # 分類問題の損失関数は基本的に「クロスエントロピー誤差関数」を用いる。
    optimizer = optim.SGD(
        params=updated_params, lr=0.001, momentum=0.9
    )  # 引数のparamsには再学習させる最終層のパラメータしか渡さない(全体を学習させたい場合は「params=net.parameters()」)

    train_model(net, data_loaders_dict, criterion, optimizer, num_epoch)


if __name__ == "__main__":
    main()