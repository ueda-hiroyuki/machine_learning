import os
import urllib.request
import zipfile
import tarfile


HOME_DIR = "src/sample_data/pytorch_advanced/2_objectdetection"


def main():

    # VOC2012のデータセットをここからダウンロード
    data_dir = f"{HOME_DIR}/data/"
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    target_path = f"{data_dir}/VOCtrainval_11-May-2012.tar"

    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)

    with tarfile.open(f"{data_dir}/VOCtrainval_11-May-2012.tar", "r:*") as t:
        t.extractall(path=f"{data_dir}/images/VOCdevkit")

    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # weights_dir = f"{HOME_DIR}/weights/"
    # if not os.path.exists(weights_dir):
    #     os.mkdir(weights_dir)

    # # target_path = f"{data_dir}/VOCtrainval_11-May-2012.tar"
    # tar = tarfile.TarFile(target_path)  # tarファイルを読み込み
    # tar.extractall(f"{HOME_DIR}/images")  # tarを解凍
    # tar.close()  # tarファイルをクローズ

    # # 学習済みのSSD用のVGGのパラメータをフォルダ「weights」にダウンロード
    # url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
    # target_path = os.path.join(weights_dir, "vgg16_reducedfc.pth")

    # if not os.path.exists(target_path):
    #     urllib.request.urlretrieve(url, target_path)

    # # 学習済みのSSD300モデルをフォルダ「weights」にダウンロード
    # url = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"
    # target_path = os.path.join(weights_dir, "ssd300_mAP_77.43_v2.pth")

    # if not os.path.exists(target_path):
    #     urllib.request.urlretrieve(url, target_path)


if __name__ == "__main__":
    main()