import numpy as np
import pandas as pd



DATA_DIR = "src/sample_data/Kaggle/classifer_cloud_images"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"

def main():
    train_data = pd.read_csv(TRAIN_PATH)
    sub_data = pd.read_csv(SAMPLE_PATH)

    print(train_data)


if __name__ == "__main__":
    main()