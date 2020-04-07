import pandas as pd
import numpy as np

# ダウンキャストすることによってdfの容量を約半分にする
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int32","int64"]]
    df[float_cols] = df[float_cols].astype("float32")
    df[int_cols] = df[int_cols].astype("int16")
    return df


def main():
    sales_train = pd.read_csv("src/sample_data/Kaggle/predict_future_price/sales_train.csv")
    df = downcast_df(sales_train)
    print(df)
    sales_by_item_id = sales_train.pivot_table(
        index=['item_id'],
        values=['item_cnt_day'], 
        columns='date_block_num', 
        aggfunc=np.sum, 
        fill_value=0
    ).reset_index()
    print(sales_by_item_id)
    sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
    print(sales_by_item_id)


if __name__ == "__main__":
    main()