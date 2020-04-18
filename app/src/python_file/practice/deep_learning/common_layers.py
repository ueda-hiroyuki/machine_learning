import numpy as np
from python_file.practice.deep_learning.common_func import CommonFunctions

cm = CommonFunctions()

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = cm.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = cm.softmax(x)
        self.loss = cm.cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.nask = None

    def forward(self, x, train_flag=True):
        if train_flag: # 学習時のみDropoutを使用する
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio #randomマスクで出力値を制限してやる(次層への伝播は行われない)
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask # 逆伝播時も順伝播時のmaskを用いる

# Affineレイヤからの出力を正規化(標準化)を行った後、活性化関数を適応する
# 内部共変量シフトの抑制
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # テスト時に使用する平均と分散   
        self.running_mean = running_mean
        self.running_var = running_var  

        # 逆伝播時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flag=True):
        self.input_shape = x.shape
        if x.ndim != 2: # 2次元配列以上(入力データが複数ある時⇒ミニバッチを考慮)
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self._forward(x, train_flag)
        return out.reshape(*self.input_shape)

    def _forward(self, x, train_flag):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flag:
            mu = x.mean(axis=0) # 平均
            xc = x - mu
            var = np.mean(xc**2, axis=0) # 分散(標準偏差**2)
            std = np.sqrt(var + 10e-7) # 標準偏差(+10e-7はゼロ除算を回避するため)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc, self.std, self.xn = xc, std, xn
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            # 学習時に算出した平均と分散から標準化を行う
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta  
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        dx = self._backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def _backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx