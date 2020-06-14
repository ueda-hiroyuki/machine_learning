import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import np_utils

SAVE_DIR = "src/sample_data/AI_jobcolle"

LABELS = np.array([
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
NUM_CLASS = 10

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        from IPython.display import clear_output
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        clear_output(wait=True)
    
    def on_train_end(self, logs=None):
        print("[Callback]on_train_end")
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.loss, label="loss")
        plt.plot(self.x, self.val_loss, label="val_loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x, self.val_acc, label="val_acc")
        plt.legend()
        plt.savefig(f"{SAVE_DIR}/train_loss_and_acc.png")

def main():
    # X_train: 32*32の3チャネル(RGB)画像が50000枚、X_test: 32*32の3チャネル(RGB)画像が10000枚
    # 扱いやすいように正規化しておく
    (X_train, y_train),(X_test, y_test) = cifar10.load_data() 
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    Y_train = np_utils.to_categorical(y_train, 10) # ターゲットをone_hot_vector化しておく
    Y_test = np_utils.to_categorical(y_test, 10) # ターゲットをone_hot_vector化しておく

    # # cifarの画像を確認
    # pos = 1
    # plt.figure(figsize=(15, 20))
    # for tgt_num in range(NUM_CLASS):
    #     tgt_idx = []
    #     for img_num in range(len(y_train)):
    #         if y_train[img_num][0] == tgt_num:
    #             tgt_idx.append(img_num)
    #     np.random.shuffle(tgt_idx)
    #     for num in tgt_idx[:10]:
    #         plt.subplot(10,10, pos)
    #         plt.imshow(X_train[num])
    #         plt.axis("off")
    #         plt.title(LABELS[tgt_num])
    #         pos+=1
    # plt.savefig(f"{SAVE_DIR}/cifar_10.samples.png")

    models = Sequential() # 各レイヤを繋げていくことができる。
    
    # block1: 畳み込み ⇒ プーリング ⇒ バッチ正規化
    
    models.add(Conv2D(16, (3,3), padding="same", input_shape=(32,32,3), kernel_initializer="he_normal")) # フィルタ数：32、フィルタサイズ：(3,3)、入力サイズと出力サイズが同じ、最初の層である場合は"input_shape"を指定
    models.add(Activation('relu')) # 活性化関数にはrelu関数を使用
    models.add(Conv2D(16, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(Conv2D(16, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(MaxPool2D(pool_size=(2, 2))) # プーリングにより画像サイズを半分にする
    models.add(BatchNormalization(momentum=0.9))
    # models.add(Dropout(0.5)) # ドロップアウトすることで過学習抑制

    # block2: 畳み込み ⇒ プーリング ⇒ バッチ正規化
    models.add(Conv2D(32, (3,3), padding="same", kernel_initializer="he_normal")) 
    models.add(Activation('relu'))
    models.add(Conv2D(32, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(Conv2D(32, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(MaxPool2D(pool_size=(2, 2))) 
    models.add(BatchNormalization(momentum=0.9))

    # block3: 畳み込み ⇒ プーリング ⇒ バッチ正規化
    models.add(Conv2D(64, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(Conv2D(64, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(Conv2D(64, (3,3), padding="same", kernel_initializer="he_normal"))
    models.add(Activation('relu'))
    models.add(MaxPool2D(pool_size=(2, 2))) 
    models.add(BatchNormalization(momentum=0.9))

    models.add(GlobalAveragePooling2D()) # 特徴量マップのサイズを1*1にして、更に1次元にする
    # models.add(Flatten()) # 3次元データを1次元データに変換 ⇒ out_h * out_w * フィルタ数 = ニューロン数
    # models.add(Dense(512, kernel_initializer="he_normal")) # 全結合層：512個のニューロン
    # models.add(Activation('relu'))
    # models.add(Dropout(0.5))
    models.add(Dense(NUM_CLASS, activation="softmax")) # 10クラス分類であるため最終出力数は10(softmax関数で確率を算出)

    adam = Adam() # 勾配降下法のアルゴリズム
    models.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["acc"]) # compileでsubmit
    models.summary() # ネットワークの確認

    # 学習
    batch_size = 1000
    epochs = 20 # 重みの更新を何回行うのか
    validation_split = 0.2 # データから評価用データとして使われる割合
    callback_plot = PlotLosses()
    es_cb =EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')

    history = models.fit(
        X_train, 
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[callback_plot, es_cb],
        validation_split=validation_split
    )

    # モデル構造の保存
    open(f"{SAVE_DIR}/models.json", "w").write(models.to_json())

    # 学習済みの重みを保存
    models.save_weights(f"{SAVE_DIR}/weight.hdf5")

    # 保存したモデル構造の読み込み
    models = model_from_json(open(f"{SAVE_DIR}/models.json", 'r').read())
    models.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc']) # 読みだしたモデルはコンパイルする必要がある

    # 保存した学習済みの重みを読み込み
    models.load_weights(f"{SAVE_DIR}/weight.hdf5")

    # 評価
    score = models.evaluate(X_test, Y_test, verbose=0)
    print("#####################score####################")
    print(f"Test loss : {score[0]}")
    print(f"Test accuracy : {score[1]}")
    print("##############################################")

    #predict_classesで画像のクラスを予想する
    img_pred=models.predict_classes(X_test)

    #5x5枚の画像を表示する
    plt.figure(figsize=(10,10))
    for i in range(25):
        rand_num=np.random.randint(0,10000)
        cifar_img=plt.subplot(5,5,i+1)
        plt.imshow(X_test[rand_num])
        #x軸の目盛りを消す
        plt.tick_params(labelbottom='off')
        #y軸の目盛りを消す
        plt.tick_params(labelleft='off')
        #画像の予想
        plt.title('pred:{0}, ans:{1}'.format(LABELS[img_pred[rand_num]], LABELS[y_test[rand_num]]))
    plt.savefig(f"{SAVE_DIR}/predicted_samples.png")

    

if __name__ == "__main__":
    main()
