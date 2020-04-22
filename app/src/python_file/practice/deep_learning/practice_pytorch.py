import logging
import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import numpy as np
import sklearn.preprocessing as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split

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

        # 畳込層+活性化関数+プーリング層+(バッチ正規化)を一塊にする
        self.block1 = nn.Sequential(
            nn.Cov2d
        )

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
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    network = NeuralNetwork()
    optimizer = op.SGD(network.parameters(), lr=0.01) # 更新手法の選択(SGD:確率的勾配降下法), parameters()はnn.Moduleのパラメータ
    criterion = nn.MSELoss() # 損失関数の定義(平均二乗誤差：二乗L2ノルム)
    
    # 3000イテレーション分回す
    for i in range(3000):
        logging.info(f'start {i}th iteration !!')
        optimizer.zero_grad() # 保持している勾配パラメータ(誤差)の初期化
        output = network(x_train) # nn.Moduleにはcall関数が定義されている(x_trainはcallの引数)
        loss = criterion(output, y_train) # 損失関数に出力値と教師データを引数として与える。
        loss.backward() # 勾配の計算(出力値-教師データ)
        optimizer.step() # パラメータの更新(勾配計算後に呼び出せる)

    # 評価
    x_test = torch.from_numpy(x_test).float()
    test_output = network(x_test) # テストデータをforwardに通す(確率分布が返ってくる)
    _, predicted = torch.max(test_output.data, 1) # 行方向の最大indexを返す
    y_predicted = predicted.numpy() # tensor型 ⇒ numpy.array型
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.sum(y_test == y_predicted) * 100 / len(y_test)
    logging.info(f'accuracy is [ {accuracy} % ]')



    

if __name__ == "__main__":
    train_nn_by_pytorch()