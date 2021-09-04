import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    两种方式 使用封装好的 feature_columnes 
        1. keras.model 加一层 DenseFeatures -> 自定义 estimator
        2. model to estimator
            这里的 bug 没处理好
"""

train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

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
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column, # 列名
                vocab)  # 词典
        ))

"""处理连续数据"""
for numeric_column in numeric_columns:
    """连续特征可以直接输入，所以直接封装"""
    feature_columns.append(tf.feature_column.numeric_column(numeric_column, dtype=tf.float32))


def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


train_dataset = make_dataset(train_df, y_train, epochs=100)
eval_dataset = make_dataset(eval_df, y_eval, epochs=1, shuffle=False)

"""
    进入 keras 模型
    第一层为 经过处理的 特征列
"""
model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns=feature_columns),
    keras.layers.Dense(100, activation="sigmoid"),
    keras.layers.Dense(100, activation="sigmoid"),
    keras.layers.Dense(2, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

"""
    两种选择
        1. 直接 model.fit()       (之前的做法)
        2. model -> estimator -> train  (bug)
"""

print("==" * 20, " model.fit")
history = model.fit(train_dataset, validation_data=eval_dataset,
                    steps_per_epoch=20,
                    validation_steps=8,
                    epochs=95)

estimator = keras.estimator.model_to_estimator(model)

"""
    input_fn -> function
    function return
        a. (features, labels)
        b. dataset -> (feature, labels)
"""

print("==" * 20, " estimator")
estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))
