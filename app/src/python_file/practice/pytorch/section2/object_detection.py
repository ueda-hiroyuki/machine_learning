import os.path as osp
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


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
    "dining" "table",
    "potted" "plant",
    "sofa",
    "tvmonitor",
]


class Anno_xml2list:
    """
    1枚の画像データに対するxml形式のアノテーションファイルを、
    画像サイズで規格化してからリスト形式に変換する
    ・input：出力クラスのリスト
    """

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, height, width):
        # 1枚の画像内のアノテーション情報を返す([[xmin, ymin, xmax, ymax, class_idx], ...])。
        ret = []
        root = ET.parse(xml_path).getroot()
        pts = ["xmin", "ymin", "xmax", "ymax"]
        for obj in root.iter("object"):  # 画像内のアノテーションの個数分だけ回す

            # difficultが"1"のものは画像では判断つかないもの
            difficult = obj.find("difficult").text
            if difficult == "1":
                continue

            label = obj.find("name").text
            xmlbox = obj.find("bndbox")  # element型
            bboxes = []
            for pt in pts:
                pixel = int(xmlbox.find(pt).text) - 1  # VOCは左上の原点が(1,1)であるため(0,0)に変更
                # バウンディングボックスの座標の規格化(入力画像のサイズに依存しないようにするため)
                if (pt == "xmin") or (pt == "xmin"):
                    pixel /= width
                else:
                    pixel /= height  # "ymin","ymax"の時は高さで規格化する
                bboxes.append(pixel)
            label_idx = self.classes.index(label)
            bboxes.append(label_idx)

            ret.append(bboxes)
        print(ret)
        return ret


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


def main_run():
    (
        train_img_path_list,
        valid_img_path_list,
        train_anno_path_list,
        valid_anno_path_list,
    ) = make_data_path_list(f"{HOME_DIR}/data/VOC2012")
    an = Anno_xml2list(voc2012_classes)
    anno_list = an(train_anno_path_list[0], 100, 100)


if __name__ == "__main__":
    main_run()