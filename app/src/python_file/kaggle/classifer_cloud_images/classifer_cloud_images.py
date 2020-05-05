import os
import logging
import random
import cv2
import numpy as np
import pandas as pd
import torch.optim as op
import albumentations as albu # 画像データ拡張ライブラリ
import segmentation_models_pytorch as smp # pytorch版セグメンテーション用ライブラリ
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from glob import glob
from catalyst import dl # pytorch用のフレームワークの１つ(学習周りのコードを簡略化する)
from torch.utils.data import TensorDataset, DataLoader, Dataset
from catalyst.contrib.nn.criterion.dice import BCEDiceLoss
from python_file.kaggle.classifer_cloud_images.functions import *
from python_file.kaggle.classifer_cloud_images.cloud_dataset import CloudDataset
from sklearn.model_selection import train_test_split
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback, AUCCallback, CheckpointCallback, InferCallback


logging.basicConfig(level=logging.INFO)

DATA_DIR = "src/sample_data/Kaggle/classifer_cloud_images"
LOG_DIR = f"{DATA_DIR}/logs/segmentation"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"
SAVE_PATH = f"{DATA_DIR}/submission.csv"
ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax"
batch_size = 16
num_workers = 0
num_epochs = 1


def gen_image(train_data):
    fig = plt.figure()
    train_images = glob(f"{DATA_DIR}/train_images/*.jpg")
    rm = random.choice(train_images).split('/')[-1]
    img_df = train_data[train_data["img_id"] == rm]
    #img_df = train_data[train_data["img_id"] == '8242ba0.jpg']
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
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    test_ids = sub_data['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    
    """
    segmentation_models_pytorchを用いて任意のモデルの事前学習を行う。
    Unetは「エンコーダ」と「デコーダ」で構成されており、今回はresnet50モデルの一部を画像特徴量抽出(エンコーダ)として用い、
    デコーダでは特徴量を元にラベリングを行う。
    """
    model = smp.Unet(
        encoder_name = ENCODER, # resnet50のモデルを事前学習させる。
        encoder_weights = ENCODER_WEIGHTS, # ImageNetで事前学習させたモデルを用いる。
        classes = 4, # 最終出力数
        activation = ACTIVATION, # 多値分類なのでsoftmax関数
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS) # 事前学習時に用いた前処理パラメータ、関数等を取得する
    
    train_dataset = CloudDataset(df=train_data, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train_data, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # model, criterion, optimizer
    optimizer = op.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2}, 
        {'params': model.encoder.parameters(), 'lr': 1e-3},  
    ])
    # ReduceLROnPlateau: 指標値(例えばloss)が○○回連続で改善しない場合は学習率を減少させる。
    scheduler = op.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = BCEDiceLoss(eps=1.)
    # SupervisedRunner：学習させるモデルやその他関数、指標値などを渡すだけ(教師ありモデルでのランナー)
    runner = dl.SupervisedRunner()
    # 学習開始
    runner.train(
        model = model, # 学習させるモデル(今回は事前学習済みのresnet50モデル)
        criterion = criterion, # 損失関数
        optimizer = optimizer, # 重みパラメータ更新手法
        scheduler = scheduler, # 学習率の減衰
        loaders = loaders, # 学習用と評価用、それぞれのDataloaderが定義されたオブジェクト
        # DiceCallback：評価関数の1つ, EarlyStoppingCallback: 指標値が改善されなくなった場合に学習を停止する
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=LOG_DIR,
        num_epochs=num_epochs,
        verbose=True,
    )
    # 推論
    encoded_pixels = []
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525)) # 確率マップ
    for i, (batch, output) in enumerate(tqdm(zip(
            valid_dataset, runner.callbacks[0].predictions["logits"]))): # tqdm：プログレスバー表示
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability

    class_params = {}
    for class_id in tqdm(range(4)):
        print("##################################")
        print(f"class_id : {class_id}")
        print("##################################")
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])


        attempts_df = attempts_df.sort_values('dice', ascending=False)
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        
        class_params[class_id] = (best_threshold, best_size)

    # 予測
    test_dataset = CloudDataset(
        df = sub_data,
        datatype = "test",
        img_ids = test_ids,
        transforms = get_validation_augmentation(),
        preprocessing = get_preprocessing(preprocessing_fn), 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = 8,
        shuffle = False,
        num_workers = num_workers,
    )
    loaders = {
        "test": test_loader
    }
    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm(loaders["test"])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for i, batch in enumerate(runner_out):
            for probability in batch:
                
                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1

    sub_data['EncodedPixels'] = encoded_pixels
    sub_data.to_csv(SAVE_PATH, columns=['Image_Label', 'EncodedPixels'], index=False)
        


if __name__ == "__main__":
    main()