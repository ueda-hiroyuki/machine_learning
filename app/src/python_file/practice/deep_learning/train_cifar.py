import os
import logging
import joblib
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
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


class LeNet(nn.Module): # LeNetは畳み込み層、プーリング層が各3層ずつの構造である
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )
        self.full_conn = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=500), # 全結合層⇒線形変換:y=x*w+b(in_featuresは直前の出力ユニット(ニューロン)数, out_featuresは出力のユニット(ニューロン)数)
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=num_classes) # out_features:最終出力数(分類クラス数)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), 64 * 4 * 4)
        out = self.full_conn(x)
        return out


def show_cifer(data, classes, path):
    H = 10
    W = 10
    fig = plt.figure(figsize=(H, W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.4, wspace=0.4)
    for i, (images, labels) in enumerate(data, 0):
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
    network_path = "src/sample_data/cifar/convolution_network.pkl"
    path = "src/sample_data/cifar"
    batch_size = 100
    epochs = 5
    num_classes = 10

    # torchvisionのtransformsには画像変換系の関数が入っている。
    transform = transforms.Compose([ # Compose関数に変換式を与える
        transforms.Resize((32,32)),
        transforms.ToTensor(), # PIL画像またはnumpy.ndarrayをTensor型に変換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizeは正規化(カラー画像は3つのチャネルを持ち、第一引数：平均、第二引数：標準偏差を与える)
    ])
    train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)
    
    # show_cifer(test_batch, IMAGE_LABELS, f'{path}/cifar_img2.png')
    network = LeNet(num_classes)
    optimizer = op.Adagrad(network.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []

    # 学習
    if not os.path.exists(network_path):
        network.train()
        for epoch in range(1, epochs+1):
            logging.info(f'===== START {epoch}th epoch !! =====')
            count_train = 0
            for i, (data, label) in enumerate(train_batch):
                optimizer.zero_grad()
                output = network(data)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output, axis=1)
                predicted, label = predicted.numpy(), label.numpy()
                cnt_train = np.sum(predicted == label)
                count_train += cnt_train
                if i % 100 == 0:
                    logging.info(f'{i}th iteration ⇒ loss: {loss.item()}')
            train_accuracy = count_train / len(train_dataset) * 100
            train_acc_list.append(train_accuracy) 

            count_test = 0
            for i, (data, label) in enumerate(test_batch):
                test_output = network(data)
                _, predicted = torch.max(test_output, axis=1)
                predicted, label = predicted.numpy(), label.numpy()
                cnt_test = np.sum(predicted == label)
                count_test += cnt_test
            test_accuracy = count_test / len(test_dataset) * 100
            test_acc_list.append(test_accuracy)    
        joblib.dump(network, network_path)
    else:
        network = joblib.load(network_path) 


    # 評価
    network.eval()
    count = 0
    for i, (data, label) in enumerate(test_batch):
        test_output = network(data)
        _, predicted = torch.max(test_output, axis=1)
        predicted, label = predicted.numpy(), label.numpy()
        cnt = np.sum(predicted == label)
        count += cnt
    final_accuracy = count / len(test_dataset) * 100
    logging.info(f'========= Finaly accuracy is {final_accuracy} !! =========')
    # if len(train_acc_list) != 0 and len(test_acc_list) != 0:
    #     accu_dict = {
    #         "epoch": range(1, epochs+1),
    #         "train": train_acc_list,
    #         "test": test_acc_list,
    #     }
    #     plt.figure()
    #     x = accu_dict['epoch']
    #     y = accu_dict['train']
    #     plt.plot(x, y, label='train')
        
    #     y = accu_dict['test']
    #     plt.plot(x, y, label='test')
    #     plt.legend()
        
    #     plt.xlabel('epoch')
    #     plt.ylabel('accuracy')
    #     plt.xlim(0, epochs)
    #     plt.ylim(0, 100)
    #     plt.savefig(f'{path}/accuracy_gragh.png')

    # 推論
    horse_img = Image.open('src/sample_data/cifar/horse.jpg')
    bird_img = Image.open('src/sample_data/cifar/bird.jpg')
    cat_img = Image.open('src/sample_data/cifar/cat.jpg')
    dog_img = Image.open('src/sample_data/cifar/dog.jpg')
    dog2_img = Image.open('src/sample_data/cifar/dog2.jpg')
    dog3_img = Image.open('src/sample_data/cifar/dog3.jpg')

    img = transform(dog3_img) 
    img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    result = network(img)
    _, predicted = torch.max(result, axis=1)
    print(IMAGE_LABELS)
    print(result)
    print(IMAGE_LABELS[predicted.numpy()[0]])



if __name__ == "__main__":
    train_cifar_by_cnn()