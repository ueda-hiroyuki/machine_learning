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
data_dir = "src/sample_data/images/hymenoptera_data"
logging.basicConfig(level=logging.INFO)


def load_data(batch_size, data_dir):
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
    
    return train_data, train_batch, val_data, val_batch 


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


def train_model(train_dataset, model, loss_func, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(1, epochs):
        logging.info('#############################')
        logging.info(f'START {epoch}th epoch !!')
        logging.info('#############################')
        for i, (data, label) in enumerate(train_dataset):
            logging.info(f'START {i}th iteration !!')
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step() # optimizer更新後に学習率を更新する。
            logging.info(f'LOSS is {loss.item()}!!')
    return model

# ファインチューニングは事前学習されたモデルの重みパラメータを使用して、全体の重みを再学習させる手法
def fine_turning():
    batch_size = 10
    train_data, train_batch, val_data, val_batch = load_data(batch_size, data_dir)
    
    # inputs, classes = next(iter(train_batch))
    # out = torchvision.utils.make_grid(inputs, nrow=5)
    # print(inputs)
    # imshow(out, title=[c for c in classes.numpy()])

    # 事前学習モデルとして「ResNet18」を用いる
    pretrained_model = models.resnet18(pretrained=True)
    num_ftrs = pretrained_model.fc.in_features # in_features: 結合層のLinearレイヤのパラメータの一つ(inputの数)
    pretrained_model.fc = nn.Linear(num_ftrs, 2) # 出力層の次元を変更する

    # 最適化するパラメータとしてはネットワークを構成するすべてのレイヤのパラメータを渡す
    optimizer = op.SGD(pretrained_model.parameters(), lr=0.01, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    # schedulerはパラメータ更新ごとに学習率を任意の関数で変化させることができる為、より効率よく勾配を収束させることができる
    # 以下の式は7エポック毎に学習率を0.1倍するという意味
    scheduler = op.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(train_batch, pretrained_model, loss_func, optimizer, scheduler, 10)


# 転移学習は、事前学習したネットワークの重みパラメータは固定し新たに追加したレイヤ(分類器層)の重みのみを再学習させる手法
def transfer_learning():
    batch_size = 10
    train_data, train_batch, val_data, val_batch = load_data(batch_size, data_dir)
    pretrained_model = models.resnet18(pretrained=True)
    for param in pretrained_model.parameters():
        # requires_grad = False とすることで事前学習モデルのすべての層の重みパラメータを固定し、これ以上学習を進めないことを明示する
        param.requires_grad = False 
    # 最終層(分類器層)に2クラス分類層を追加する(新たに追加されたレイヤのrequires_gradはデフォルトでTrue)
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss() # 損失関数の定義

    # 最適化するパラメータとして引数で与えるのは、新たに追加した最終レイヤのパラメータのみ
    optimizer = op.SGD(pretrained_model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = op.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(train_batch, pretrained_model, criterion, optimizer, scheduler, 5)
    visualize_model(model, val_batch)

def visualize_model(model, num_images=10):
    class_index = json.load(open('src/sample_data/Image-Classifier/data/imagenet_class_index.json', 'r'))
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
 
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_batch):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
 
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_index[preds[j]]))
                plt.savefig(f'src/sample_data/images/samples/predicted_hymenoptera{j}.jpg')
 
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
 


if __name__ == "__main__":
    transfer_learning()