import os
import logging
import random
import numpy as np
import pandas as pd
import albumentations as albu # 画像データ拡張ライブラリ
import segmentation_models_pytorch as smp # pytorch版セグメンテーション用ライブラリ
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from catalyst import dl # pytorch用のフレームワークの１つ
from torch.utils.data import TensorDataset, DataLoader, Dataset
from python_file.kaggle.classifer_cloud_images.functions import *
from python_file.kaggle.classifer_cloud_images.cloud_dataset import CloudDataset
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/classifer_cloud_images"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"


def gen_image(train_data):
    fig = plt.figure()
    train_images = glob(f"{DATA_DIR}/train_images/*.jpg")
    rm = random.choice(train_images).split('/')[-1]
    img_df = train_data[train_data["img_id"] == rm]
    #img_df = train_data[train_data["img_id"] == '8242ba0.jpg']
    print(img_df)
    img = Image.open(f"{DATA_DIR}/train_images/{rm}")
    for idx, (_, row) in enumerate(img_df.iterrows()):
        img_id, label_name = row["img_id"], row["label"]
        plt.imshow(img)
        pixel = row["EncodedPixels"]
        # labelがNanの場合もある
        try:
            mask = rle_decode(pixel)
        except:
            # 画像(およびマスク)は1400 x 2100であり、予測マスクは350x525である。
            # label未定義の場合はゼロ配列を作成
            mask = np.zeros((1400, 2100))
        plt.imshow(mask, alpha=0.5, cmap='gray')
        plt.title(f"ID: {img_id}, Label: {label_name}")
        plt.savefig(f"{DATA_DIR}/masked_image{idx}.jpg")


def main():
    train_data = pd.read_csv(TRAIN_PATH)
    sub_data = pd.read_csv(SAMPLE_PATH)

    train_data["img_id"] = train_data["Image_Label"].apply(lambda x: x.split("_")[0])
    train_data["label"] = train_data["Image_Label"].apply(lambda x: x.split("_")[1])
    # gen_image(train_data)
    
    # 1つの画像あたりlabelをいくつ含んでいるのかをカウントする
    id_mask_count = train_data.loc[train_data['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    # 学習データを学習用と検証用に分割(画像1枚あたりが含んでいるラベル数が偏らないようにtrain_test_splitの引数としてstratifyを与えている。)
    train, valid = train_test_split(id_mask_count["img_id"], random_state=42, test_size=0.1, stratify=id_mask_count["count"])
    test = sub_data["Image_Label"].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = "softmax"

    model = smp.Unet(
        encoder_name = ENCODER, # resnet50のモデルを事前学習させる。
        encoder_weights = ENCODER_WEIGHTS, # ImageNetで事前学習させたモデルを用いる。
        classes = 4, # 最終出力数
        activation = ACTIVATION, # 多値分類なのでsoftmax関数
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS) # 事前学習時に用いた前処理パラメータ、関数等を取得する
    print(preprocessing_fn)

    


if __name__ == "__main__":
    main()