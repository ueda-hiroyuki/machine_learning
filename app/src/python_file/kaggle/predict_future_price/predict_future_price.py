import matplotlib.pyplot as plt
import pandas as pd
import typing as t
import numpy as np
import lightgbm as lgb
from datetime import datetime

DATA_DIR = "src/sample_data/Kaggle/predict_future_price"
SALES_TRAIN_PATH = f"{DATA_DIR}/sales_train.csv"
ITEMS_PATH = f"{DATA_DIR}/items.csv"
ITEM_CATEGORIES_PATH = f"{DATA_DIR}/item_categories.csv"
PRED_PRICE_PATH = f"{DATA_DIR}/pred_price.csv"
SHOPS_PATH = f"{DATA_DIR}/shops.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"


def preprocessing(train, items, item_categories, pred_price, shops, test):
    train = train[train.item_price > 0].reset_index(drop = True)
    train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0

    train.loc[train.shop_id == 0, "shop_id"] = 57
    test.loc[test.shop_id == 0 , "shop_id"] = 57
    train.loc[train.shop_id == 1, "shop_id"] = 58
    test.loc[test.shop_id == 1 , "shop_id"] = 58
    train.loc[train.shop_id == 11, "shop_id"] = 10
    test.loc[test.shop_id == 11, "shop_id"] = 10
    train.loc[train.shop_id == 40, "shop_id"] = 39
    test.loc[test.shop_id == 40, "shop_id"] = 39
    shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
    shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"

    return train, items, item_categories, pred_price, shops, test


def main():
    train = pd.read_csv(SALES_TRAIN_PATH)
    items = pd.read_csv(ITEMS_PATH)
    item_categories = pd.read_csv(ITEM_CATEGORIES_PATH)
    pred_price = pd.read_csv(PRED_PRICE_PATH)
    shops = pd.read_csv(SHOPS_PATH)
    test = pd.read_csv(TEST_PATH)

    train, items, item_categories, pred_price, shops, test = preprocessing(train, items, item_categories, pred_price, shops, test)
    category = []
    for cat in shops.category.unique():
        print(cat, len(shops[shops.category == cat]) )
        if len(shops[shops.category == cat]) > 4:
            category.append(cat)
    shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" ) # 母数の多いカテゴリはそのまま、それ以外を「etc(その他)」としている。
    print(shops)


if __name__ == "__main__":
    # 2015年11月の売り上げを予測する。
    main()