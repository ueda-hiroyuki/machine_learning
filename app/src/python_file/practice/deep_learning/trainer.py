import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from collections import OrderedDict
from python_file.practice.deep_learning.test_learning_technique import *


# ニューラルネットワークの学習を行うクラス
class Trainer:
    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epoch_num=20,
        batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": 0.01},
        evaluate_sample_num_per_epoch=None, 
        verbose=True,
    ):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.verbose = verbose
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # パラメータ更新時の手法
        optimizer_class_dict = {
            'sgd': SGD,
            'momentum': Momentum,
            'adagrad': Adagrad
        } 

        self.optimizer = optimizer_class_dict[optimizer](**optimizer_param)
        self.train_size = x_train.shape[0] # 入力データ数(全画像枚数)
        self.iter_per_epoch = max(self.train_size / batch_size, 1) # 全学習データを何回のバッチで網羅できるか 
        self.max_iter = int(epoch_num * self.iter_per_epoch) # イテレーション数
        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        for i in range(self.max_iter):
            self.train_step() # イテレーション数だけ学習を行う。

        test_acc = self.network.accuracy(self.x_test, self.t_test)
        logger.info(f'=============== Final Test Accuracy : {test_acc}===============')

    
    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_train_batch = self.x_train[batch_mask]
        t_train_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_train_batch, t_train_batch)

        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_train_batch, t_train_batch)
        self.train_loss_list.append(loss)
        logging.info(f"train loss : {loss}")

        # 特定のイテレーションにおいて評価を行う
        if self.current_iter % self.iter_per_epoch == 0:
            sel.current_epoch += 1
            # 全データ
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            # 1epoch毎に評価を行う場合
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(f"=== epoch: {self.current_epoch}, train acc: {train_acc}, test acc: {test_acc} ===")
        self.current_iter += 1
