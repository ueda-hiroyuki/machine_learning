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

        # すべてのBboxに対し、現在のBbox=indexがiと被っている 値まで設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])  # clampで最小値に制限
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, min=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, min=y2[i])

        # w, h のtensorサイズをindexを1つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clampした状態のBboxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さがマイナスになっている者は0に変更
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態のBboxの面積を求める
        inter = tmp_h * tmp_w

        # IoUを算出する((intersect)/(area_a + area_b - intersect))
        rem_area = torch.index_select(area, 0, idx)  # 元のBboxの面積
        union = (rem_area - inter) + area[i]  # 2つのエリアの和(OR)の面積
        iou = inter / union

        # iouがoverlap値よりも大きいものは削除(小さいものは残す)
        idx = idx[
            iou.le(overlap)
        ]  # tensor.le: less than or equal to (指定値より低いものはTrueを返す)

    return keep, count
