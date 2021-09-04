import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    使用 feature_columnes 封装
        稀疏数据可以 很方便的 one-hot 编码
        连续数据可以 变成离散特征
"""

train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

"""
    根据 数据类型 对 特征 进行分类
"""

categorical_columns = {"sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"}

numeric_columns = {"age", "fare"}

feature_columns = []

"""处理离散数据"""
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()  # .unique()返回有哪些不同的值

    """
        tf.feature_column.indicator_column()
            对离散特征列进行 one-hot 编码
            
        tf.feature_column.categorical_column_with_vocabulary_list(key, list)
            生成特征列
    """

    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(categorical_column, vocab)))

"""处理连续数据"""
for numeric_column in numeric_columns:
    """连续特征可以直接输入，所以直接封装"""
    feature_columns.append(tf.feature_column.numeric_column(numeric_column, dtype=tf.float32))


"""生成数据集"""
def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


train_dataset = make_dataset(train_df, y_train, batch_size=5)

"""查看生成的数据集"""
for x, y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_column = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_column)(x).numpy())

for x, y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())
