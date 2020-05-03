import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from python_file.kaggle.classifer_cloud_images.functions import *
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
    
    print(train_data)
    # 1つの画像あたりlabelをいくつ含んでいるのかをカウントする
    id_mask_count = train_data.loc[train_data['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    print(id_mask_count)
    # 学習データを学習用と検証用に分割(画像1枚あたりが含んでいるラベル数が偏らないようにtrain_test_splitの引数としてstratifyを与えている。)
    train, valid = train_test_split(id_mask_count["img_id"], random_state=42, test_size=0.1, stratify=id_mask_count["count"])
    print(train, valid)


if __name__ == "__main__":
    main()