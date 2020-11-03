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
from modules import *


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
        nn.init.constant_(self.weight, self.scale)  # weightの値がすべてscale(=20)になる

    def forward(self, x):
        # norm: チャネル方向の2乗和の平方根
        norm = x.pow(2).sum(dim=1, keepdim=True).np.sqrt() + self.eps
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
        self.num_priors = len(cfg["feature_maps"])  # sourceの個数(=6)
        self.steps = cfg["steps"]  # DBoxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"]  # 小さい正方形Dboxのサイズ
        self.max_sizes = cfg["max_sizes"]  # 大きい正方形Dboxのサイズ
        self.aspect_ratio = cfg["aspect_ratio"]  # 長方形Dboxのアスペクト比

    def generate_dbox_list(
        self,
    ):
        """
        Dboxのリストを作成する関数
        """
        mean = []
        for n, f in enumerate(self.feature_maps):  # feature_maps: (38,19,10,5,3,1)
            for i, j in product(range(f), repeat=2):
                # 特徴量マッピングの画像サイズ
                f_k = self.image_size / self.steps[n]  # self.steps[n]: 8,16,32,64,

                # dboxの中心座標x,y (但し、0~1で規格化している)
                # 1 pixel毎にbboxを作成するため、+0.5すると1*1の中心部分に相当する
                # f_k(特徴量マップのサイズ)で割ることで中心座標を0~1に規格化している
                cx = (j + 0.5) / f_k  # dboxの中心x座標
                cy = (i + 0.5) / f_k  # dboxの中心y座標

                # アスペクト比(1,1)の小さいdbox(cx, cy, width, height)
                # min_sizes: [30,60,111,162,213,264]
                s_k = self.min_sizes[n] / self.image_size  # Dboxの1辺に長さ
                mean.append([cx, cy, s_k, s_k])

                # アスペクト比(1,1)の大きいdbox(cx, cy, width, height)
                s_k_prime = np.sqrt(s_k * (self.max_sizes[n] / self.image_size))
                mean.append([cx, cy, s_k_prime, s_k_prime])

                # その他アスペクト比のDboxたち
                for ar in self.aspect_ratio[n]:  # ex)1,2,3,1/2,1/3
                    mean.append(
                        [cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)]
                    )  # ボックスの幅および高さを逆にする
                    mean.append([cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)])

        output = torch.Tensor(mean).view(-1, 4)  # view：第2引数として設定した値x ⇒ (〇行, x列)に自動変換
        output = torch.clamp(output, max=1, min=0)  # dboxが画像からはみ出さないようにmin maxでのクリップ
        return output


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]

        # SSDのネットワークを作成する
        self.vgg = generate_vgg()  # vggモジュールを作成
        self.extra = generate_extra()  # extraモジュールを作成
        self.L2Norm = L2Norm()  # L2Normクラス
        self.loc, self.conf = generate_loc_and_conf(
            num_classes=cfg["num_classes"],
            bbox_aspect_num=cfg["bbox_aspect_num"],
        )

        # dboxの作成
        dbox = DefaultBox(cfg)
        self.dbox_list = dbox.generate_dbox_list()

        # 推論時には"Detect"クラスを使用
        if self.phase == "interface":
            self.detect = Detect()
