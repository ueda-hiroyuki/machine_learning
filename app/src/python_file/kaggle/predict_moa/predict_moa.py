import pandas as pd
import numpy as np


DATA_DIR = "src/sample_data/Kaggle/predict_moa"


def run_all():
    train_raw = pd.read_csv(f"{DATA_DIR}/train_features.csv")
    test_raw = pd.read_csv(f"{DATA_DIR}/test_features.csv")
    train_target_nonscored_raw = pd.read_csv(f"{DATA_DIR}/train_targets_nonscored.csv")
    train_target_scored_raw = pd.read_csv(f"{DATA_DIR}/train_targets_scored.csv")

    print(train_raw, test_raw, train_target_nonscored_raw, train_target_scored_raw)


if __name__ == "__main__":
    run_all()