import numpy as np
import pandas as pd
import joblib

PKL_PATH = 'src/sample_data/outlier.pkl'


def create_outlier_pkl(df: pd.DataFrame, sigma: int, path: str) -> None:
    sd_dic = {}
    for c in df.columns:
        series = df[c]
        average = np.mean(series)
        sd = np.std(series)
        outlier_min = average - (sd)*sigma
        outlier_max = average + (sd)*sigma
        sd_dic[c] = (outlier_min, outlier_max)
    joblib.dump(sd_dic, path)

def remove_outlier(df: pd.DataFrame, path: str) -> pd.DataFrame:
    sd_dic = joblib.load(path)
    for k, v in sd_dic.items():
        series = df[k]
        df[k] = series[(series > v[0]) & (series < v[1])]
    print(df)


def main() -> None:
    array = np.arange(100)
    array = np.array([i if i%11!=0 else i+10000 for i in array])
    df = pd.DataFrame(
        array.reshape(25,4),
        columns=['col_0', 'col_1', 'col_2', 'col_3']
    )
    print(df)
    create_outlier_pkl(df, 2, PKL_PATH)
    df = remove_outlier(df, PKL_PATH)




if __name__ == "__main__":
    main()