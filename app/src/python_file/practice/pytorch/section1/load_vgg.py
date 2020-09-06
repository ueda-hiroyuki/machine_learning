import numpy as np
import json
import os
import urllib.request
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

DATA_DIR = "src/sample_data/pytorch_advanced/1_image_classification/data"

# 入力画像の前処理クラス
class BasePreprocessor:
    """
    画像のサイズをリサイズし、色の標準化を行うクラス

    ・input
        resize：リサイズ先の画像サイズ
        mean：(R, G, B) ⇒ 各色チャネルの平均値
        std：(R, G, B) ⇒ 各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose(  # Compose: データをロードした後に行う下処理の関数を構成する
            [
                transforms.Resize(resize),  # 短い辺の長さがrisizeの値となる
                transforms.CenterCrop(resize),  # 画像中央をrisize*risizeで切り取り
                transforms.ToTensor(),  # Torchテンソルに変換
                transforms.Normalize(mean, std),  # 色情報の標準化
            ]
        )

    def __call__(self, img):
        return self.base_transform(img)


class ILSVRCPredictor:
    """
    ILSVRCデータに対するモデルの出力からラベルを求めるクラス
    ・input
        class_index: 辞書型で定義されたラベル
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_label(self, out):
        """
        予測確率が最大のラベル名を出力する
        ・input
            out: vgg16からの出力(torch.Size([1,1000]))

        """
        max_id = np.argmax(
            out.detach().numpy()
        )  # Tensor型のデータをNumpy型に変換する(detach：netからの切り離し)
        predicted_label = self.class_index[str(max_id)][
            1
        ]  # max_idに対応したラベルをclass_indexから取得する

        return predicted_label


def main():
    ILSVRC_class_index = json.load(open(f"{DATA_DIR}/imagenet_class_index.json", "r"))
    predictor = ILSVRCPredictor(ILSVRC_class_index)

    use_pre_trained = True  # 学習済みのパラメータを使用する
    net = models.vgg16(pretrained=use_pre_trained)  # 学習済みのvgg16モデルのインスタンスを作成する。
    net.eval()  # 推論モードに変更

    img_path = f"{DATA_DIR}/goldenretriever-3724972_640.jpg"
    img = Image.open(img_path)

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = BasePreprocessor(resize, mean, std)
    img_transformed = transform(img)  # call関数実行 ⇒ return (3,224,224)
    # img_transformed = img_transformed.transpose((1, 2, 0))  # transpose to (224,224,3)
    # img_transformed = np.clip(img_transformed, 0, 1)  # 数値を指定の範囲に規格化する
    inputs = img_transformed.unsqueeze_(
        0
    )  # reshape to torch.Size[(1, 3, 224, 224)] ← この形にしないとモデルは検知できない

    # モデルに入力
    out = net(inputs)

    predicted_label = predictor.predict_label(out)

    print(predicted_label)


if __name__ == "__main__":
    main()
