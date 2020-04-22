import logging
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import numpy as np
import sklearn.preprocessing as sp
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sample_data.deep_learning_documents.dataset.mnist import load_mnist


logging.basicConfig(level=logging.INFO)

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        """
        Convolutional Neural Network
        ネットワーク構成：
            Input - CONV - CONV - MaxPool - CONV - CONV - MaxPool - FullConected - Output
            ※MaxPoolの直後にバッチ正規化を実施
        引数：
            num_classes: 分類するクラス数（＝出力層のユニット数）
        """
        # 親クラスとインスタンス自身をsuper()に渡す
        super(ConvolutionalNeuralNetwork, self).__init__() # 親クラスの継承(nn.Moduleのinit()を実行する)

        # 畳込層+活性化関数+プーリング層+(バッチ正規化)を一塊にする ⇒ Sequentialにはコンストラクタで渡された順番で順次追加されていく
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # 1枚の画像に対し、16個のフィルタ(チャンネル)を適応
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1), # 1枚の画像に対し、16個のフィルタ(チャンネル)を適応
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),# 出力サイズ: チャネル=16, 高さ=27, 幅=27
            nn.BatchNorm2d(16) # inputされるチャネルの数(N,C,H,W)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # 1枚の画像に対し、32個のフィルタ(チャンネル)を適応
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # 1枚の画像に対し、32個のフィルタ(チャンネル)を適応
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # 出力サイズ: チャネル=32, 高さ=26, 幅=26
            nn.BatchNorm2d(32) # inputされるチャネルの数(N,C,H,W)
        )
        self.full_conn = nn.Sequential(
            nn.Linear(in_features=32*26*26, out_features=512), # 全結合層⇒線形変換:y=x*w+b(in_featuresは直前の出力ユニット(ニューロン)数, out_featuresは出力のユニット(ニューロン)数)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=num_classes) # out_features:最終出力数(分類クラス数)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), 32 * 26 * 26) # block2までの出力は3次元であるため、Affineを行う為には2次元変換する
        output = self.full_conn(x)
        return output # 最終出力は2次元


# input_size:1(n:4), hidden_size:2(n:10,8), output_size: 1(3)のシンプルなニューラルネットワーク
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() # nn.Moduleを継承
        # Affineレイヤの定義
        self.Affine1 = nn.Linear(4,10) # Linear層は全結合
        self.Affine2 = nn.Linear(10, 8)
        self.Affine3 = nn.Linear(8, 3)

    # 順伝播　
    def forward(self, x):
        out1 = f.relu(self.Affine1(x))
        out2 = f.relu(self.Affine2(out1))
        out3 = self.Affine3(out2)
        return out3 

def train_nn_by_pytorch():
    iris = datasets.load_iris()
    label = iris.target.reshape(len(iris.target),1)
    # 1次元ラベルをone_hot_vectorに変換する
    one_hot_label = sp.OneHotEncoder(sparse=False).fit_transform(label)
    # 学習データとテストデータを分割する
    x_train, x_test, y_train, y_test = train_test_split(iris.data, one_hot_label, test_size=0.25)

    # numpy.arrayをpytorchで扱える形に変換
    x_train = torch.tensor(x_train, dtype=torch.float32) 
    y_train = torch.tensor(y_train, dtype=torch.float32)

    network = NeuralNetwork()
    optimizer = op.SGD(network.parameters(), lr=0.01) # 更新手法の選択(SGD:確率的勾配降下法), parameters()はnn.Moduleのパラメータ
    criterion = nn.MSELoss() # 損失関数の定義(平均二乗誤差：二乗L2ノルム)
    
    # 3000イテレーション分回す
    for i in range(3000):
        logging.info(f'start {i}th iteration !!')
        optimizer.zero_grad() # 保持している勾配パラメータ(誤差)の初期化 ⇒ 勾配がイテレーション毎に加算されてしまうため
        output = network(x_train) # nn.Moduleにはcall関数が定義されている(x_trainはcallの引数)
        loss = criterion(output, y_train) # 損失関数に出力値と教師データを引数として与える。
        loss.backward() # 勾配の計算(出力値-教師データ)
        optimizer.step() # パラメータの更新(勾配計算後に呼び出せる)

    # 評価
    x_test = torch.tensor(x_test, dtype=torch.float32)
    test_output = network(x_test) # テストデータをforwardに通す(確率分布が返ってくる)
    _, predicted = torch.max(test_output.data, 1) # 行方向の最大indexを返す
    y_predicted = predicted.numpy() # tensor型 ⇒ numpy.array型
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.sum(y_test == y_predicted) * 100 / len(y_test)
    logging.info(f'accuracy is [ {accuracy} % ]')
    print(network)


def train_cnn_by_pytorch(): 
    batch_size = 100
    num_classes = 10
    epochs = 3

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    network = ConvolutionalNeuralNetwork(num_classes)
    
    # データのフォーマットを変換：PyTorchでの形式 = [画像数，チャネル数，高さ，幅]
    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test.reshape(10000, 1, 28 ,28)
 
    # PyTorchのテンソルに変換
    x_train = torch.Tensor(x_train).float()
    x_test = torch.Tensor(x_test).float()
    t_train = torch.LongTensor(t_train) # labelにはint型(LongTensor型)を用いる ⇒ floatは×
    t_test = torch.LongTensor(t_test)

    # 学習用と評価用のデータセットを作成(60000枚分)
    train_dataset = TensorDataset(x_train, t_train)
    test_dataset = TensorDataset(x_test, t_test)
    
    # データセットをバッチサイズ毎に分割する(ex: dataset:100, batchsize:20 ⇒ tensor(20 × 5)に分割)
    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # パラメータ更新手法の定義
    optimizer = op.Adagrad(network.parameters(), lr=0.01, lr_decay=0, weight_decay=0.05, initial_accumulator_value=0, eps=1e-10)

    # 損失関数の定義
    loss_func = nn.CrossEntropyLoss() # CrossEntropy誤差はone_hot_vectorに対応していない

    network.train()
    # 学習(エポック数3回 ⇒ パラメータは随時更新)
    for i in range(1, epochs+1):
        logging.info(f'===== START {i}th epoch !! =====')
        for i, (data, label) in enumerate(train_batch):
            optimizer.zero_grad()
            output = network(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logging.info(f'{i}th iteration ⇒ loss: {loss.item()}')





    

if __name__ == "__main__":
    train_cnn_by_pytorch()