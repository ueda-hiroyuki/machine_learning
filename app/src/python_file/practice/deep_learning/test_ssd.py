import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from glob import glob
from sample_data.pytorch_handbook.chapter7.ssd import build_ssd
from matplotlib import pyplot as plt

WEIGHT_PATH = 'src/sample_data/pytorch_handbook/chapter7/weights/ssd300_mAP_77.43_v2.pth'
IMAGES_PATH = 'src/sample_data/images/*.jpg'
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def detect(network, image, count):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (300, 300)).astype(np.float32)  # 300*300にリサイズ
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)  # [300,300,3]→[3,300,300]
    xx = x.unsqueeze(0)     # [3,300,300]→[1,3,300,300]  
    
    # 順伝播を実行し、推論結果を出力    
    y = network(xx)
    detections = y.data
  
    plt.figure(figsize=(10,6))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()
    # 推論結果をdetectionsに格納
    
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    # バウンディングボックスとクラス名を表示
    for i in range(detections.size(1)):
        j = 0
        # 確信度confが0.6以上のボックスを表示
        # jは確信度上位200件のボックスのインデックス
        # detections[0,i,j]は[conf,xmin,ymin,xmax,ymax]の形状
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = VOC_CLASSES[i-1]
            display_txt = f'{label_name}: {score}'
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            print(pt)
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
    # plt.savefig(f'src/sample_data/images/detected_image{count}.jpg')
    # plt.close()
    
    
def test_ssd():    
    # 学習済みSSDモデルを読み込み
    network = build_ssd('test', 300, 21)   
    network.load_weights(WEIGHT_PATH)

    images = glob(IMAGES_PATH)
    count = 1
    for idx, image in enumerate(images):
        image = cv2.imread(image, cv2.IMREAD_COLOR)  
        detect(network, image, count)
        count += 1


if __name__ == "__main__":
    test_ssd()