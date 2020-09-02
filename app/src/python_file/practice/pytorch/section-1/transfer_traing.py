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


def main():
    img = Image.open(f"{DATA_DIR}/goldenretriever-3724972_640.jpg")

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, "train")
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)

    print(img_transformed)

    plt.imshow(img_transformed)
    plt.savefig(f"{DATA_DIR}/Augumentations.jpg")


if __name__ == "__main__":
    main()