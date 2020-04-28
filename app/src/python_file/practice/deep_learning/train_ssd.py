import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sample_data.pytorch_handbook.chapter7.ssd import build_ssd
from sample_data.pytorch_handbook.chapter7.data import *

args = {'dataset':'BCCD',  # VOC → BCCD
    'basenet':'vgg16_reducedfc.pth',
    'batch_size':12,
    'resume':'',
    'start_iter':0,
    'num_workers':0,  # 4 → 0
    'cuda':True,  # Macの場合False
    'lr':5e-4,
    'momentum':0.9,
    'weight_decay':5e-4,
    'gamma':0.1,
    'save_folder':'weights/'
    }
weight_path = ""


def main():
    cfg = voc
    dataset = VOCDetection(root=VOC_ROOT,transform=SSDAugmentation(cfg['min_dim'],MEANS))


if __name__ == "__main__":
    main() 