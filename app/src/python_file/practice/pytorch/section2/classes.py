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
from common import *


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
        self.num_classes = cfg["num_classes"]  # クラス数：21

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

    def forward(self, x):
        sources = list()  # loc, confへ入力するsource1~6を格納する
        loc = list()  # locの出力を格納する
        conf = list()  # confの出力を格納する

        for i in range(23):  # vggの23層目まで実行し、source1を出力する
            x = vgg[i](x)

        # 23層目の出力をL2Normに入れ、source1とする
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vggを最後まで計算させ、それをsource2として格納
        for i in range(23, len(self.vgg)):
            x = vgg[i](x)
        sources.append(x)

        # extraモジュールの計算を行いsource3~6を出力
        for n, v in enumerate(self.extra):
            x = nn.ReLU(v(x), inplace=True)  # 畳み込み後自前で活性化関数に入れ込む
            if n % 2 == 1:
                sources.append(x)

        # source1~6のそれぞれに対応する畳み込み層をloc, confモジュールから引っ張る
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(
                l(x).permute(0, 2, 3, 1).contiguous()
            )  # contiguous(): メモリ上で要素配列を整える
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # locとconfの型を以下に変更
        # loc：Tensor([batch_num], 8732*(アスペクト比の種類数))
        # conf：Tensor([batch_num], 8732*(class数))
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # locとconfの型を更に整える
        # loc：Tensor([batch_num], 8732, (アスペクト比の種類数))
        # conf：Tensor([batch_num], 8732, (class数))
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 最後に出力する
        output = (loc, conf, self.dbox_list)

        # 学習時はそのままoutput
        if self.phase == "train":
            return output
        else:
            # 推論時は推論結果をoutput
            return self.detect(output[0], output[1], output[2])


class Detect(torch.autograd.Function):
    """
    SSDの推論時にconfとlocの出力から、被りを除去したBboxを返す
    torch.autograd.Functionを継承し、クラス内でforward, backword関数を定義することで独自のautograd演算子を定義することができる
    """

    def __init__(self, conf_threshold=0.01, top_k=200, nms_threshold=0.45):
        self.softmax = nn.Softmax(dim=-1)  # confをsoftmax関数で正規化する
        self.conf_threshold = conf_threshold  # confが0.01以上のDboxのみを扱う
        self.top_k = top_k  # non_maxmum_supressionでconfが高い上位〇個を計算に使用する
        self.nms_threshold = nms_threshold  # non_maxmum_supression内でIOUが〇より大きいBboxは削除

    def forward(self, loc, conf, dbox_list):
        """
        順伝播の計算を実行する
        ・input
            ⇒ loc：オフセット情報
            ⇒ conf：検出の信頼度
            ⇒ dbox_list：Dboxのリスト[8732,4]
        ・outtput
            ⇒ torch.Size([batch_num, 21, confのtop_k個分, BBox情報])
        """

        # 各サイズを取得
        num_batch = loc.size(0)  # バッチサイズ
        num_dbox = loc.size(1)  # Dboxの数(8732個)
        num_classes = loc.size(2)  # クラス数

        # confはsoftmaxを用いて正規化する
        conf = self.softmax(conf)

        # 出力の型を作成する[batch_num, 21, confのtop_k個分, BBox情報]
        output = torch.zero(num_batch, num_classes, self.top_k, 5)

        # confの列を並び替える
        conf_pred = conf.transpose(2, 1)

        # ミニバッチごとのループ処理
        for i in range(num_batch):
            # decodeを用いてlocとDbox情報からBboxを求める
            decoded_boxes = decode(loc[i], dbox_list)

            # confのコピーを作成
            conf_score = conf_pred[i].clone()

            # クラスごとのループ処理(背景クラス(idx0)は処理しないため1から)
            for cl in range(1, num_classes):
                # conf_score:[21, 8732] ⇒ 各クラスのscoreをDbox毎に持っている
                cs = conf_score[cl]
                c_mask = cs.gt(
                    self.conf_threshold
                )  # 信頼度が〇以上のものは1, それ以外は0となる(gt：greater_than)
                scores = cs[c_mask]  # 閾値を超えたBboxの個数となる

                if scores.nelement() == 0:  # nelement:要素の個数を算出する ⇒ つまりscoreが[]の場合は何もしない
                    continue

                # c_mask([8732])をdecoded_scoreにも適応できるようにリサイズ
                l_mask = c_mask.unsqueeze(1).expand_as(
                    decoded_boxes
                )  # tensor([8732, 4])

                # decoded_boxes[l_mask]で1次元になってしまうため、viewでtensor([閾値以上のBbox数, 4])とする
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # non_maximum_supressionを実施し、被っているBboxは除去する
                # ids：confの降順にnon_maximum_supressionを通過したBboxのindexリスト
                # count：non_maximum_supressionを通過したBboxの数
                ids, count = non_maximun_supression(
                    boxes, scores, self.nms_threshold, self.top_k
                )

                # outputにBbox結果を格納する
                out[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1
                )
        return output  # tensor([1, 21, 200, 5]) ⇒ 1枚毎のBbox情報(non_maximum_supression済)
