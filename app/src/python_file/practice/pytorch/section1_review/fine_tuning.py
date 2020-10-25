import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from transfer_training import *

DATA_DIR = "src/sample_data/pytorch_advanced/1_image_classification/data"


def get_training_params(net):
    # 学習させるparameterたち
    update_params1 = []
    update_params2 = []
    update_params3 = []

    # 学習させるparameterの名前
    update_params_name1 = ["features"]  # featuresモジュールはすべて再学習
    update_params_name2 = [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
    ]  # classifierモジュール内の0,3,6層目は全結合層
    update_params_name3 = [
        "classifier.6.weight",
        "classifier.6.bias",
    ]

    # 各層のparameterをlistに追加していく
    for name, param in net.named_parameters():
        if name in update_params_name1:
            param.requires_grad = True
            update_params1.append(param)
        elif name in update_params_name2:
            param.requires_grad = True
            update_params2.append(param)
        elif name in update_params_name3:
            param.requires_grad = True
            update_params3.append(param)
        else:
            param.requires_grad = False

    return update_params1, update_params2, update_params3


def main():
    # 学習用の画像データ群のパスを取得
    train_path_list = gen_datapath_list(phase="train")
    valid_path_list = gen_datapath_list(phase="valid")

    batch_size = 30
    epoch_nums = 3
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Datasetの作成
    transformer = ImageTransformer(resize, mean, std)
    train_dataset = HymenopteraDataset(
        path_list=train_path_list, transform=transformer, phase="train"
    )
    valid_dataset = HymenopteraDataset(
        path_list=valid_path_list, transform=transformer, phase="valid"
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

    dataloader_dict = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    # 学習済みvgg16モデルを読み込み
    net = load_vgg16()
    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()
    update_params1, update_params2, update_params3 = get_training_params(net)

    # 最適化手法の決定(parameter毎に学習時のハイパラを変える)
    optimizer = optim.SGD(
        [
            {"params": update_params1, "lr": 1e-4, "momentum": 0.9},
            {"params": update_params2, "lr": 5e-4, "momentum": 0.9},
            {"params": update_params3, "lr": 1e-3, "momentum": 0.9},
        ]
    )

    # 学習・検証の実施
    for name, param in net.named_parameters():
        if (name == "classifier.6.weight") or name == "classifier.6.bias":
            print(param)
    train_model(net, dataloader_dict, criterion, optimizer, epoch_nums)
    for name, param in net.named_parameters():
        if (name == "classifier.6.weight") or name == "classifier.6.bias":
            print(param)

    # 学習済みのNNの重みとバイアスを保存
    torch.save(net.state_dict(), f"{DATA_DIR}/weight_fine_tuned.pth")


if __name__ == "__main__":
    main()