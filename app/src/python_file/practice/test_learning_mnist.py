import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as t
from sample_data.deep_learning_documents.dataset import mnist as mn


def load_data():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train


def mean_squared_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
    e = 1/2 * (np.sum((y-k)**2))
    return e


def cross_entropy_error(y: t.Sequence, t: t.Sequence) -> t.Sequence:
    delta = 1e-7
    e = - np.sum(t * np.log(y + delta)) # +deltaはlog0(-inf)とならないような対策
    return e


def main():
    batch_size = 10
    x_train, t_train = load_data()
    train_size = x_train.shape[0]
    
    batch_mask = np.random.choice(train_size, batch_size) 
    print(batch_size, train_size, batch_mask)


if __name__ == "__main__":
    main()