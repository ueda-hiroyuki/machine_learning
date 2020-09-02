import numpy as np
import json
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


def main():
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
    img_transformed = img_transformed.numpy().transpose(
        (1, 2, 0)
    )  # transpose to (224,224,3)
    img_transformed = np.clip(img_transformed, 0, 1)  # 数値を指定の範囲に規格化する
    print(img_transformed)


if __name__ == "__main__":
    main()
