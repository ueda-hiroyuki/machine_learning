import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "src/sample_data/pytorch_advanced/1_image_classification/data"


class ImageTransformer:
    """
    画像をリサイズし、色の標準化を行うクラス(学習時と推論時でデータの前処理を変える)
    imput:
        ・resize：リサイズ後の画像のサイズ
        ・mean(R,G,B)：各色チャネルの平均値
        ・std(R,G,B)：各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(  # Composeを使用することでチェーンさせた前処理が簡単に書ける(前処理パイプライン的な)
                [
                    transforms.RandomResizedCrop(
                        resize, scale=(0.5, 1)
                    ),  # データの水増し(指定したスケールで画像のリサイズをしたり、アスペクト比を変更。最終的にはresizeの値で出力)
                    transforms.RandomHorizontalFlip(),  # 50%の確率で画像の上下を反転する。
                    transforms.ToTensor(),  # Tensor型に変換
                    transforms.Normalize(mean, std),  # 指定した色情報に標準化
                ]
            ),
            "valid": transforms.Compose(  # Composeを使用することでチェーンさせた前処理が簡単に書ける(前処理パイプライン的な)
                [
                    transforms.Resize(resize),  # リサイズ
                    transforms.CenterCrop(resize),  # 中央部のトリミング
                    transforms.ToTensor(),  # Tensor型に変換
                    transforms.Normalize(mean, std),  # 指定した色情報に標準化
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        """
        input
            ・img：画像データ
            ・phase：学習フェーズなら"train"、評価フェーズなら"valid"を指定
        """
        return self.data_transform[phase](img)  # transforms.Composeにimgを入れるのを忘れずに。


class HymenopteraDataset(Dataset):
    """
    学習、評価用のデータセット生成クラス
    ・input：
        ・path_list: 学習用もしくは評価用の画像データpathが入ったリスト
        ・transform: 前処理実行クラスのインスタンス
        ・phase: "train" or "valid"
    """

    def __init__(self, path_list, transform=None, phase="train"):
        self.path_list = path_list  # 画像のファイスパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase

    def __len__(self):
        # 画像の枚数を返す
        return len(self.path_list)

    def __getitem__(self, idx):
        """
        前処理をした画像のTensor形式のデータとラベルを取得する
        ・input：idx(画像の順番)
        """

        # 指定したindexの画像を読み込む
        img_path = self.path_list[idx]
        img = Image.open(img_path)  # shape: (height, width, RGB)

        # 画像の前処理を実施
        preprocessed_img = self.transform(
            img, self.phase
        )  # shape: torch([3, 224, 224])

        # 画像のラベルを取得し、数値に変換
        label = img_path[84:88]  # pathの文字列から抽出
        if label == "ants":
            label = 0
        else:
            label = 1

        return preprocessed_img, label  # 前処理後の画像データと、正解ラベル(数字)


def gen_datapath_list(phase="train"):
    return glob(f"{DATA_DIR}/hymenoptera_data/{phase}/**/*.jpg")


def load_vgg16():
    # 学習済みのvgg16モデルを読み込む
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(
        in_features=4096, out_features=2
    )  # デフォルトでは出力ユニットは1000個であるため2つのユニットに置換する
    net.train()  # 学習モードに設定
    return net


def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    """
    モデルを学習させる関数
    ここでは1エポック毎に学習と検証を交互に実施する。
    """
    for epoch in range(num_epochs):
        print("#################################")
        print(f"start {epoch+1}/{num_epochs} epoch !!")
        print("#################################")
        for phase in ["train", "valid"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0  # モデルの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確認するため、epochが0の時の学習は省略
            if (epoch == 0) and (phase == "train"):
                continue

            # DataLoaderからミニバッチを取り出す
            loader = dataloader_dict[phase]
            for iteration, (inputs, labels) in enumerate(loader):
                print(f"{iteration}th iteration")
                optimizer.zero_grad()  # iteration毎(parameter更新毎)に勾配を初期化

                # 順伝播計算(forward)
                with torch.set_grad_enabled(  # 勾配計算のOn/Off
                    phase == "train"
                ):  # 勾配計算(順伝播)は学習時しか行わない(requires_grad=True)
                    outputs = net(inputs)  # 出力層はsoftmax関数なので0,1である確率が出力される。
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  # 確率が大きいほうのindexを取得
                    if phase == "train":
                        loss.backward()  # 学習時には誤差逆伝播を行う(正解との誤差を元に、中間層の重みを修正する)
                        optimizer.step()  # パラメータの更新(勾配計算後に呼び出せる)

                    # 1epoch毎に損失を計算
                    # loss.item: ミニバッチの損失の平均、inputs.size: batch_sizeと同等
                    epoch_loss += loss.item() * inputs.size(
                        0
                    )  # iteration全体のloss(loss平均*バッチサイズ)
                    epoch_corrects += torch.sum(preds == labels)

            # epoch毎のlossと正解率
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)  # 画像1枚当たりの損失
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print(f"################# {phase}__{epoch+1}th_epoch ################")
            print(f"{phase} Loss:{epoch_loss}, Acc:{epoch_acc}")
            print("#################################")


def main():
    # 学習済みVGG16モデルを読み込む
    net = load_vgg16()

    # 学習用の画像データ群のパスを取得
    train_path_list = gen_datapath_list(phase="train")
    valid_path_list = gen_datapath_list(phase="valid")

    batch_size = 30
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformer = ImageTransformer(resize, mean, std)
    train_dataset = HymenopteraDataset(
        path_list=train_path_list, transform=transformer, phase="train"
    )
    valid_dataset = HymenopteraDataset(
        path_list=valid_path_list, transform=transformer, phase="valid"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )  # 学習用dataloader(画像を取り出す順番はランダム)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )  # 評価用dataloader

    dataloader_dict = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }  # 辞書型でまとめる

    params_to_update = []  # 転移学習時に学習させるパラメータを格納する
    update_param_names = [
        "classifier.6.weight",
        "classifier.6.bias",
    ]  # 今回は出力層のパラメータのみを再学習させる。
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True  # 再度学習させる層のパラメータにはTrue
            params_to_update.append(param)
        else:
            param.requires_grad = False  # パラメータを変えない層はFalse

    criterion = nn.CrossEntropyLoss()  # 分類問題なので損失関数はクロスエントロピー
    optimizer = optim.SGD(
        params=params_to_update, lr=0.001, momentum=0.9
    )  # 最適化手法の決定(paramsには学習させたい層のparameterのみを与える)

    # 学習・検証の実施
    num_epochs = 3
    train_model(net, dataloader_dict, criterion, optimizer, num_epochs)


if __name__ == "__main__":
    main()
