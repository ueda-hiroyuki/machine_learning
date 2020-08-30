import pandas as pd
import numpy as np

DATA_DIR = '.sample_data/Kaggle/predict_winner'
TRAIN_PATH = f'{DATA_DIR}/tain_data.csv'
TEST_PATH = f'{DATA_DIR}/test.csv'

def main():
    train_raw_data = pd.read_csv(TRAIN_PATH)
    test_raw_data = pd.read_csv(TEST_PATH)
    print(train_raw_data)


if __name__ == '__main__':
    main()