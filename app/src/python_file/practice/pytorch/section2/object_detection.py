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
from classes import *


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

cfg_ssd = {
    "num_classes": 21,  # 20クラス+背景(1クラス)
    "input_size": 300,  # 画像の入力サイズ
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDboxのアスペクト比の種類
    "feature_maps": [38, 19, 10, 5, 3, 1],  # 特徴量マッピング(各source)のサイズ
    "steps": [8, 16, 32, 64, 100, 300],  # dboxの大きさ
    "min_sizes": [30, 60, 111, 162, 213, 264],  # dboxの大きさ
    "max_sizes": [60, 111, 162, 213, 264, 315],  # dboxの大きさ
    "aspect_ratio": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}


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


def decode(loc, dbox_list):
    """
    オフセット情報(loc)を使用し、DboxをBboxに変換する
    ・input
    　　⇒ loc: SSDモデルで推論するオフセット情報(tensor([8732,4])) ⇒ [Δcx, Δct, Δwidth, Δheight]
    　　⇒ dbox_list: デフォルトボックスの情報(tensor([8732,4])) ⇒ [cx_d, cy_d, width_d, height_d]
    ・output
    　　⇒ bboxes: バウンディングボックスの情報(tensor([8732,4])) ⇒ (xmin,ymin,xmax,ymax)
    """

    # オフセット情報からbboxを求める
    boxes = torch.cat(
        (
            dbox_list[:, :2] * (1 + 0.1 * loc[:, :2]),
            dbox_list[:, 2:] * torch.exp(0.2 * loc[:, 2:]),
        ),
        dim=1,
    )  # (cx, cy, w, h)
    boxes[:, :2] -= boxes[:, 2:] / 2  # (xmin, ymin)を算出
    boxes[:, 2:] += boxes[:, :2]  # (xmin, ymin)にw,hをそれぞれ足して(xmax, ymax)を算出

    return boxes  # (xmin,ymin,xmax,ymax) ⇒ tensor(8732,4)


def non_maximun_supression(boxes, scores, overlap=0.45, top_k=200):
    """
    non_maximun_supression
    　⇒ 1つの画像に対してBboxが何個も重なってしまうことが発生する。その中でも信頼度(confidence)が高いもののみを選択する手法
    　　 Bboxが被っているもの(overlap>0.45)を削除する

    ・input
    　⇒ boxes：bbox情報
    　⇒ score：信頼度情報(confidence)
    ・output
    　⇒ count：nms後の残ったBboxの個数
    　⇒ keep：nms後に残ったBboxのindex番号のリスト
    """

    # returnのひな型を作成する
    count = 0
    keep = (
        scores.new(scores.size(0)).zero_().long()
    )  # score(tensor型)と同じTensorを作成する(要素はすべて0であり、long()で整数)

    # 各Bboxの面積areaを求める。
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    area = torch.mul(x2 - x1, y2 - y1)  # Bboxの面積(縦*横)

    # boxesをコピーする
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    v, idx = scores.sort(
        0, descending=True
    )  # 降順に並べ替える ⇒ v: 並び替えた後のtensor、idx: どう並べ替えたのかの情報
    idx = idx[:top_k]  # 上位〇番目までをピックアップ

    while idx.numel() > 0:  # idxの要素数が0(Bboxが存在しない)場合はループしない
        i = idx[0]  # confの最大index
        keep[count] = i  # keepに最大confのBboxを入れる
        count += 1

        if idx.size(0) == 1:  # 最後のBboxの場合はループを抜ける
            break

        idx = idx[1:]  # confが最大のものは除去(keepに入っているため)

        # keepに格納したBboxと被りが大きいものを抽出して削除する
        # 1つ減らしたidxまでのBboxを選択(outに選択されたBboxが入る)
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)


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
    dbox = DefaultBox(cfg_ssd)
    dbox_list = dbox.generate_dbox_list()
    ssd = SSD(phase="train", cfg=cfg_ssd)
    aa = dbox_list.new()


if __name__ == "__main__":
    main_run()