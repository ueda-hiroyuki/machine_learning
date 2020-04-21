import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as f
import sklearn.preprocessing as sp
from sklearn import datasets
from sklearn.model_selection import train_test_split


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
        print("####################x#####################")
        print(x.shape)
        print("####################out1#####################")
        print(out1.shape)
        print("#####################out2####################")
        print(out2.shape)
        print("######################out3###################")
        print(out3.shape)
        return out3 

    # 逆伝播
    def backward(self, dout):
        ...

def train_nn_by_pytorch():
    iris = datasets.load_iris()
    label = iris.target.reshape(len(iris.target),1)
    # 1次元ラベルをone_hot_vectorに変換する
    one_hot_label = sp.OneHotEncoder(sparse=False).fit_transform(label)
    # numpy.arrayをpytorchで扱える形に変換
    iris_data = torch.from_numpy(iris.data).float()
    # 学習データとテストデータを分割する
    x_train, x_test, y_train, y_test = train_test_split(iris_data, one_hot_label, test_size=0.25)

    network = NeuralNetwork()
    optimizer = op.SGD(network.parameters(), lr=0.01) # 更新手法の選択(SGD:確率的勾配降下法), parameters()はnn.Moduleのパラメータ
    loss = nn.MSELoss() # 損失関数の定義
    # 3000イテレーション分回す
    for i in range(2):
        optimizer.zero_grad() # 勾配パラメータを削除
        output = network.forward(x_train)

        # loss = criterion(output, y)
        # loss.backward()
        # optimizer.step()


    

if __name__ == "__main__":
    train_nn_by_pytorch()