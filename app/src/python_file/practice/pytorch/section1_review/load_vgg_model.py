import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

DATA_DIR = "src/sample_data/pytorch_advanced/1_image_classification/data"


class BaseTransform:
    """
    画像をリサイズし、色の標準化を行うクラス
    imput:
        ・resize：リサイズ後の画像のサイズ
        ・mean(R,G,B)：各色チャネルの平均値
        ・std(R,G,B)：各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.base_transform = (
            transforms.Compose(  # Composeを使用することでチェーンさせた前処理が簡単に書ける(前処理パイプライン的な)
                [
                    transforms.Resize(resize),  # リサイズ
                    transforms.CenterCrop(resize),  # 中央部のトリミング
                    transforms.ToTensor(),  # Tensor型に変換
                    transforms.Normalize(mean, std),  # 指定した色情報に標準化
                ]
            )
        )

    def __call__(self, img):
        return self.base_transform(img)  # 前処理させるには、インスタンスにimageを渡す必要がある。


class LabelPredictor:
    """
    NNの出力結果からラベルを推論するクラス
    imput: class_index(indexとラベルの辞書型)
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict(self, output):
        """
        NNの出力に対し、全ラベルから対応する確率が最も高いラベル名を取得する
        input：output(NNからの出力 ⇒ tensor([1,1000]))
        """
        detached = output.detach()  # tensor型をnumpy型にするために、NWから切り離す
        max_id = np.argmax(detached.numpy())  # numpy配列の中で値が最も大きいindexを取得
        label_name = self.class_index[str(max_id)][1]
        return label_name


def load_and_preprocess_image(path, resize, mean, std):
    img = Image.open(path)
    loader = BaseTransform(resize, mean, std)
    preprocessed_img = loader(img)  # tensor([3,224,224]) 224*224が3チャネル分
    return preprocessed_img


def load_vgg16():
    # 学習済みのvgg16モデルを読み込む
    # torchvision: pytorchのパッケージで、データセットや学習済みモデルなどで構成されている。

    net = models.vgg16(
        pretrained=True
    )  # vgg16を使用(学習済みパラメータを使用する場合はpretrainedを"True"にする)
    net.eval()  # 推論モードに設定
    return net


def main():
    # 学習済みのvgg16モデルを読み込む
    net = load_vgg16()

    # 画像の読み込み及び前処理動作の確認
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_path = f"{DATA_DIR}/goldenretriever-3724972_640.jpg"  # (640, 426)サイズの画像
    preprocessed_img = load_and_preprocess_image(img_path, resize, mean, std)
    input = preprocessed_img.unsqueeze(0)  # 0次元目のバッチディメンションを作成

    # ラベルの推論準備
    class_index = json.load(
        open(f"{DATA_DIR}/imagenet_class_index.json", "r")
    )  # ImageNetで与えられる1000種類のラベル群
    predictor = LabelPredictor(class_index)

    # 推論結果
    output = net(input)  # return tensor([1,1000])

    # ラベルの推論
    predicted_name = predictor.predict(output)

    print(predicted_name)


if __name__ == "__main__":
    main()