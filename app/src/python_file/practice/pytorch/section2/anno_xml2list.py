import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET


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
        return np.array(ret)