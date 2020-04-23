import os
import logging
import joblib
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

IMAGE_LABELS = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
])

def show_cifer(testloader, classes, path):
    H = 10
    W = 10
    fig = plt.figure(figsize=(H, W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.4, wspace=0.4)
    for i, (images, labels) in enumerate(testloader, 0):
        for k in range(0, images.size()[0]):
            # numpyに変換後、[3, 32, 32] -> [32, 32, 3] に変換
            numpy_array = images[k].numpy().transpose((1, 2, 0))
            plt.subplot(H, W, k+1)
            plt.imshow(numpy_array)
            plt.title("{}".format(classes[labels[k]]), fontsize=12, color = "green")
            plt.axis('off')
        break
    plt.savefig(path)


def train_cifar_by_cnn():
    path = "src/sample_data/cifar"
    batch_size = 100
    epoch = 5 

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True,download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False,download=True, transform=transform)
    
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)
    
    show_cifer(test_batch, IMAGE_LABELS, f'{path}/cifar_img.png')


if __name__ == "__main__":
