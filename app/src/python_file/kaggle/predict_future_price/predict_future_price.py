import re
import time
import matplotlib.pyplot as plt
import pandas as pd
import typing as t
import numpy as np
import lightgbm as lgb
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import product

DATA_DIR = "src/sample_data/Kaggle/predict_future_price"
SALES_TRAIN_PATH = f"{DATA_DIR}/sales_train.csv"
ITEMS_PATH = f"{DATA_DIR}/items.csv"
ITEM_CATEGORIES_PATH = f"{DATA_DIR}/item_categories.csv"
PRED_PRICE_PATH = f"{DATA_DIR}/pred_price.csv"
SHOPS_PATH = f"{DATA_DIR}/shops.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x

def preprocessing_shops(shops):
    # shopsの前処理
    shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
    shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
    category = [] # 登場回数の少ないカテゴリは"etc"とする
    for cat in shops.category.unique():
        if len(shops[shops.category == cat]) > 4:
            category.append(cat)
    shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" ) # 母数の多いカテゴリはそのまま、それ以外を「etc(その他)」としている。
    shops["category"] = LabelEncoder().fit_transform(shops.category) # categoryとcityのカテゴリ変数をencording
    shops["city"] = LabelEncoder().fit_transform(shops.city)
    shops = shops.drop("shop_name", axis=1)
    return shops

def preprocessing_item_category(item_categories):
    # item_categoryの前処理
    item_categories["type_code"] = item_categories.item_category_name.apply(lambda x: x.split()[0]).astype(str) # 文字列の" "で区切られている部分の先頭の文字列を取得する。
    item_categories.loc[(item_categories.type_code == "Игровые") | (item_categories.type_code == "Аксессуары"), "category"] = "Игры"
    category = [] # 登場回数の少ないカテゴリは"etc"とする
    for cat in item_categories.type_code.unique():
        if len(item_categories[item_categories.type_code == cat]) > 4:
            category.append(cat)
    item_categories.type_code = item_categories.type_code.apply(lambda x: x if (x in category) else "etc")
    item_categories["type_code"] = LabelEncoder().fit_transform(item_categories.type_code)
    item_categories["split"] = item_categories.item_category_name.apply(lambda x: x.split("-"))
    item_categories["subtype"] = item_categories.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    item_categories["subtype_code"] = LabelEncoder().fit_transform(item_categories.subtype)
    item_categories = item_categories.loc[:, ["item_category_id", "type_code", "subtype_code"]]
    return item_categories

def preprocessing_items(items):
    # itemsの前処理
    items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
    items["name1"], items["name3"] = items.item_name.str.split("(", 1).str
    items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items = items.fillna('0')
    items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))
    items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")
    items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
    items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
    items.loc[ items.type == "", "type"] = "mac"
    items.type = items.type.apply( lambda x: x.replace(" ", "") )
    items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
    items.loc[ items.type == 'рs3' , "type"] = "ps3"
    remove_cols = []
    for name, value in items["type"].value_counts().items():
        if value < 40:
            remove_cols.append(name) 
        else:
            pass
    items.name2 = items.name2.apply(lambda x: "etc" if (x in remove_cols) else x)
    items = items.drop(["type"], axis = 1)
    items.name2 = LabelEncoder().fit_transform(items.name2)
    items.name3 = LabelEncoder().fit_transform(items.name3)
    items.drop(["item_name", "name1"],axis = 1, inplace= True)
    return items



def preprocessing_train_test(train, test):
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
    train["revenue"] = train["item_cnt_day"] * train["item_price"]
    test["date_block_num"] = 34 # 0~33までを学習データとし用い、34(ひと月分)の売り上げを予測する
    test = test.apply(lambda x: x.astype(np.int16))

    return train, test


def gen_lag_feature(matrix, lags, cols):
    pre_cols = ["shop_id", "item_id"]
    for col in cols:
        _df = matrix.loc[:, [*pre_cols, col]]
        print(_df)
        for lag in lags:
            matrix[f"lag_{lag}"] = _df.groupby(pre_cols)[col].shift(lag)
    return matrix

# def gen_lag_feature(df, lags, cols):
#     for col in cols:
#         tmp = df[["date_block_num", "shop_id","item_id",col ]]
#         for i in lags:
#             shifted = tmp.copy()
#             shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
#             shifted.date_block_num = shifted.date_block_num + i
#             print(shifted)
#             df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
#     print(df.isna().sum())
#     return df



def main():
    train = pd.read_csv(SALES_TRAIN_PATH)
    items = pd.read_csv(ITEMS_PATH)
    item_categories = pd.read_csv(ITEM_CATEGORIES_PATH)
    pred_price = pd.read_csv(PRED_PRICE_PATH)
    shops = pd.read_csv(SHOPS_PATH)
    test = pd.read_csv(TEST_PATH)

    train, test = preprocessing_train_test(train, test)
    shops = preprocessing_shops(shops)
    item_categories = preprocessing_item_category(item_categories)
    items = preprocessing_items(items)

    matrix = []
    cols  = ["date_block_num", "shop_id", "item_id"]
    for i in range(34): # date_block_num(1月分)が0~33まで存在する。
        sales = train[train["date_block_num"] == i] # 1月毎の売り上げを抽出
        sales_matrix = np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype = np.int16) # 月、item_id、shop_idの組み合わせ(直積)を算出
        matrix.append(sales_matrix)
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols).sort_values(cols) # 月、item_id、shop_idの組み合わせをdataframeにしたもの
    matrix = matrix.apply(lambda x: x.astype(np.int16))

    group = train.groupby(["date_block_num", "shop_id", "item_id"]).agg({"item_cnt_day": ["sum"]}) # ["date_block_num", "shop_id", "item_id"]の組み合わせごとの総売り上げ数
    group.columns = ["item_cnt_month"]
    group = group.reset_index()
    merged_matrix = pd.merge(matrix, group, how="left", on=cols)
    merged_matrix["item_cnt_month"] = merged_matrix["item_cnt_month"].fillna(0).clip(0, 20) # その月で売り上げのなかったitemは0で置換し、最小値を0、最大値を20として外れ値を除去

    submit_ids = test.loc[:, "ID"]
    merged_matrix = pd.concat([merged_matrix, test.drop("ID", axis=1)], axis=0).fillna(0).reset_index(drop=True)

    # 与えられたデータフレームのマージ
    merged_items = pd.merge(
        items,
        item_categories,
        on="item_category_id",
        how="left"
    )
    merged_matrix = pd.merge(
        merged_matrix,
        merged_items,
        on="item_id",
        how="left"
    )
    merged_matrix = pd.merge(
        merged_matrix,
        shops,
        on="shop_id",
        how="left"
    )
    merged_matrix = merged_matrix.apply(lambda x: x.astype(np.int16))
    merged_matrix = gen_lag_feature(merged_matrix, [1,2,3], ["item_cnt_month"]) # 1,2,3か月前の"item_cnt_month"を特徴量に追加

    group = matrmerged_matrixix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})
    group.columns = ["date_avg_item_cnt"]
    group.reset_index(inplace = True)
    print(group)


if __name__ == "__main__":
    # 2015年11月の売り上げを予測する。
    main()