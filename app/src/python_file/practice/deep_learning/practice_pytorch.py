import torch
import torch.nn as nn
import torch.nn.functional as f

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
        super(ConvolutionalNeuralNetwork, self).__init__() # 親クラスの継承(nn.Moduleのinit()を実行する)



def main():
    ...


if __name__ == "__main__":
    main()