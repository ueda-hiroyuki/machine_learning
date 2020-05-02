import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from python_file.kaggle.classifer_cloud_images.functions import *

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/classifer_cloud_images"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"

def gen_image(train_data):
    fig = plt.figure()
    train_images = glob(f"{DATA_DIR}/train_images/*.jpg")
    rm = random.choice(train_images).split('/')[-1]
    img_df = train_data[train_data["img_id"] == rm]

    img = Image.open(f"{DATA_DIR}/train_images/{rm}")
    print()

    for idx, (_, row) in enumerate(img_df.iterrows()):
        pixel = row["EncodedPixels"]
        # labelがNanの場合もある
        try:
            mask = rle_decode(pixel)
        except:
            # 画像(およびマスク)は1400 x 2100であり、予測マスクは350x525である。
            # label未定義の場合はゼロ配列を作成
            mask = np.zeros((1400, 2100))
        print(mask.shape)

def main():
    train_data = pd.read_csv(TRAIN_PATH)
    sub_data = pd.read_csv(SAMPLE_PATH)

    train_data["img_id"] = train_data["Image_Label"].apply(lambda x: x.split("_")[0])
    train_data["label"] = train_data["Image_Label"].apply(lambda x: x.split("_")[1])
    gen_image(train_data)



if __name__ == "__main__":
    main()