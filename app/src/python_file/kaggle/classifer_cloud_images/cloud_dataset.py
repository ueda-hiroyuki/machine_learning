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
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/classifer_cloud_images"

"""
Datasetは、入力データとそれに対応するラベルを1組返すモジュール。
自前のデータを扱いたいときは自分のデータをリードして返してくれるDatasetを実装する必要がある。
Datasetクラスでは__len__と__getitem__を実装しなければならない。
⇒ __len__は、len(obj)で実行されたときにコールされる関数。
⇒ __getitem__は、obj[i]のようにインデックスで指定されたときにコールされる関数。
"""
class CloudDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame = None, 
        datatype: str = "train", 
        img_ids:np.array = None, 
        transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
        preprocessing = None,    
    ):
        self.df = df
        if datatype != "test":
            self.data_folder = f"{DATA_DIR}/train_images"
        else:
            self.data_folder = f"{DATA_DIR}/test_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_name = img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = f"{self.data_folder}/{image_name}"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         