import torch
import cv2
import torchvision
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from data_augumentation import *
from anno_xml2list import *
from modules import *


HOME_DIR = "src/sample_data/pytorch_advanced/2_objectdetection"
voc2012_classes = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]


class DataTransform:
    """
    画像とアノテーションの前処理を行うクラス(学習時と推論時は異なる挙動)
    学習時はData Augumentationを行う。
    ・input：
    　　　input_size：入力の画像サイズ
    　　　color_mean：画像の色の標準化を行う。

    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose(
                [
                    ConvertFromInts(),  # int型をfloat型に変換
                    ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                    PhotometricDistort(),  # 画像の色彩をランダムに変化させる
                    Expand(color_mean),  # 画像のキャンバスを広げる
                    RandomSampleCrop(),  # 画像内の一部をランダムに抜き出す
                    RandomMirror(),  # ランダムに画像を反転させる
                    ToPercentCoords(),  # アノテーションデータを規格化
                    Resize(input_size),  # 画像サイズを(input_size * input_size)に変更
                    SubtractMeans(color_mean),  # BGRの色の平均値を引き算
                ]
            ),
            "valid": Compose(
                [
                    ConvertFromInts(),
                    Resize(input_size),
                    SubtractMeans(color_mean),
                ]
            ),
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list  # 学習用画像データのパスのリスト
        self.anno_list = anno_list  # アノテーション情報のパスのリスト
        self.phase = phase
        self.transform = transform  # 前処理クラスのインスタンス
        self.transform_anno = transform_anno  # アノテーションの前処理クラスのインスタンス

    def __len__(
        self,
    ):
        # 画像の枚数を返す
        return len(self.img_list)

    def __getitem__(self, index):
        # 前処理をした画像データ(tensor型)とアノテーションを取得
        img, gt, _, _ = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        # 前処理をした画像データ(tensor型)とアノテーション、画像の高さと幅を取得
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channel = img.shape

        # アノテーション情報をリストにする
        anno_list = self.transform_anno(self.anno_list[index], height, width)

        # 前処理を実施
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4]
        )
        # 色チャネルをBGR⇒RGBに変更、そして(高さ、幅、チャネル数)⇒(チャネル数、高さ、幅)に変更
        img = torch.from_numpy(img[:, :, (2, 0, 1)]).permute(2, 0, 1)
        # アノテーションのpixel情報とラベルをくっつける
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, gt, height, width


class L2Norm(nn.Module):
    """
    カスタムレイヤー
    ・お約束
        ⇒ nn.Modileを継承する
        ⇒ レイヤーのparameterはnn.Prameterで設定する。
        ⇒ 順伝播はforwardメソッドを定義する。
    """

    def __init__(self, in_channels=512, scale=20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(in_channels))  # nn.Parameterの引数はTensor型
        self.scale = scale  # 係数weightの初期値として設定する値
        self.reset_parameters()  # parameterの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        # 結合パラメータの値を大きさscaleの値にする初期化を実行
        init.constant_(self.weight, self.scale)  # weightの値がすべてscale(=20)になる

    def forward(self, x):
        # norm: チャネル方向の2乗和の平方根
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)  # normで割ることで正規化

        # self.weightはtensor([512])なので、tensor([batch_num, 512, 特徴量マップHeight, 特徴量マップwidth])に変換
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x  #  重み * 正規化された入力値
        return out


class DefaultBox:
    def __init__(self, cfg):
        super(DefaultBox, self).__init__()

        # 初期設定
        self.image_size = cfg["input_size"]  # 画像のサイズ(=300)
        self.feature_maps = cfg[
            "feature_maps"
        ]  # 各sourceの特徴量mapのサイズ(38*38, 19*19, 10*10, 5*5, 3*3, 1*1)
        self.num_priors = len(cfg["feature_maps"]) # sourceの個数(=6)
        self.steps = cfg["steps"] # DBoxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"] # 小さい正方形Dboxのサイズ
        self.max_sizes = cfg["max_sizes"] # 大きい正方形Dboxのサイズ
        self.aspect_ratio = cfg["aspect_ratio"] # 長方形Dboxのアスペクト比

    def generate_dbox_list(self,):
        """
        Dboxのリストを作成する関数
        """
        mean = []
        for n, f in enumerate(self.feature_maps): # featire_maps: (38,19,10,5,3,1)
            for i, j in product(range(f), repeat=2):
                # 特徴量マッピングの画像サイズ
                f_k = self.image_size / self.steps[n] # self.steps[n]: 8,16,32,64,
                
                # dboxの中心座標x,y (但し、0~1で規格化している)
                # 1 pixel毎にbboxを作成するため、+0.5すると1*1の中心部分に相当する
                # f_k(特徴量マップのサイズ)で割ることで中心座標を0~1に規格化している
                cx = (j + 0.5) / f_k # dboxの中心x座標
                cy = (i + 0.5) / f_k # dboxの中心y座標

                # アスペクト比1の小さいdbox(cx, cy, width, height)
                




def make_data_path_list(root_dir):
    """
    dataへのpath(画像データ)を格納したlistを返す
    ・input：data_dir = f"{HOME_DIR}/data/VOC2012"
    """
    # 画像データとアノテーションデータへのpath
    img_path = osp.join(root_dir, "JPEGImages", "%s.jpg")
    anno_path = osp.join(root_dir, "Annotations", "%s.xml")

    # 学習用と検証用の画像の情報が記載されているtxtファイルを取得
    train_img_ids = osp.join(root_dir, "ImageSets", "Main/train.txt")  # txtファイルのpath
    valid_img_ids = osp.join(root_dir, "ImageSets", "Main/val.txt")  # txtファイルのpath

    train_img_path_list = []
    valid_img_path_list = []
    train_anno_path_list = []
    valid_anno_path_list = []
    for row in open(train_img_ids):  # txtファイルを開き、記載してある画像名を1行ずつ取得
        id = row.strip()
        train_img_path = osp.join(img_path % id)  # 上記"%s"の部分に"id"が入る
        train_anno_path = osp.join(anno_path % id)
        train_img_path_list.append(train_img_path)
        train_anno_path_list.append(train_anno_path)

    for row in open(valid_img_ids):  # txtファイルを開き、記載してある画像名を1行ずつ取得
        id = row.strip()
        valid_img_path = osp.join(img_path % id)  # 上記"%s"の部分に"id"が入る
        valid_anno_path = osp.join(anno_path % id)
        valid_img_path_list.append(valid_img_path)
        valid_anno_path_list.append(valid_anno_path)

    return (
        train_img_path_list,
        valid_img_path_list,
        train_anno_path_list,
        valid_anno_path_list,
    )


def od_collate_fn(batch):
    """
    datasetから取り出すアノテーションのサイズが画像により異なる。
    バッチにするときどの画像がどの矩形かを結びつける必要があり、idxを付ける必要がある。
    この変化に対応可能なDataloaderを作成する
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0]はimg
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]はannotation
    imgs = torch.stack(
        imgs, dim=0
    )  # [torch(3,300,300), torch(3,300,300), torch(3,300,300)] ⇒ torch(3, 3,300,300)
    return imgs, targets


def main_run():
    """
    SSDは１度のCNN演算で物体の「領域候補検出」と「クラス分類」の両方を行うことができる
    """
    (
        train_img_path_list,
        valid_img_path_list,
        train_anno_path_list,
        valid_anno_path_list,
    ) = make_data_path_list(f"{HOME_DIR}/data/VOC2012")

    color_mean = (104, 117, 123)
    input_size = 300
    batch_size = 3
    transform = DataTransform(input_size, color_mean)
    transform_anno = Anno_xml2list(voc2012_classes)

    # datasetの作成
    train_dataset = VOCDataset(
        train_img_path_list, train_anno_path_list, "train", transform, transform_anno
    )
    valid_dataset = VOCDataset(
        valid_img_path_list, valid_anno_path_list, "valid", transform, transform_anno
    )

    # Dataloaderを作成
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn
    )
    # 辞書型にまとめる
    dataloaders_dict = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    batch_iter = iter(dataloaders_dict["train"])
    images, targets = next(batch_iter)

    vgg = generate_vgg()
    extra = generate_extra()
    loc, conf = generate_loc_and_conf()
    print(loc, conf)


if __name__ == "__main__":
    main_run()