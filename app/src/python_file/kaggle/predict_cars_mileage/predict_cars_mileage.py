import pandas as pd
import numpy as np
import lightgbm


def main() -> None:    
    train = pd.read_table('src/sample_data/Kaggle/predict_cars_mileage/train.tsv')
    test = pd.read_table('src/sample_data/Kaggle/predict_cars_mileage/test.tsv')
    name_df = train.loc[:,'car name']
    print(list(train['horsepower']))
    train['horsepower'] = train['horsepower'].where(train['horsepower'] == '?', train['horsepower'].astype({'horsepower': float}).mean())
    print(train)
    train = train.drop('car name', axis=1).astype({'id': float, 'cylinders': float, 'horsepower': float, 'model year': float, 'origin': float})
    print(train.dtypes)
    cor_df = pd.DataFrame(np.corrcoef(train.T), columns=train.columns)
    print(cor_df)

if __name__ == "__main__":
    main()