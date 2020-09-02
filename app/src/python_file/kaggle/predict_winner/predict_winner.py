import pandas as pd
import numpy as np

DATA_DIR = 'src/sample_data/Kaggle/predict_winner'
TRAIN_PATH = f'{DATA_DIR}/train_data.csv'
TEST_PATH = f'{DATA_DIR}/test_data.csv'

def main():
    train_raw_data = pd.read_csv(TRAIN_PATH)
    test_raw_data = pd.read_csv(TEST_PATH)
    print(train_raw_data)
    print(test_raw_data)

    print(train_raw_data.info())


if __name__ == '__main__':
    main()