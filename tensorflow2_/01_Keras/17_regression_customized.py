import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing

"""
    1. 使用自定义损失函数 替换 框架内部的损失函数
    2. 使用 lambda自定义 layer
       使用子类自定义 在 '18_xxx.py'
"""

housing = fetch_california_housing()
print(housing.DESCR)
# data => 8个特征
print(housing.data.shape)
# target => 加利福尼亚地区的房屋中位价
print(housing.target.shape)

from sklearn.model_selection import train_test_split

# test_size=0.25
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_valid_scaled = scaler.transform(x_valid)

"""自定义损失函数"""


def costomized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


"""自定义 tf.nn.softplus : log(1+e^x)"""
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))
print("customized_softplus: ",customized_softplus([-10., -5., 0., 5., 10.]))

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="sigmoid", input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
    customized_softplus, # 相当于 keras.layers.Dense(1, activation="softplus")
])

# 查看 类中的 变量和 可训练变量
# layer = keras.layers.Dense(100, input_shape=[None, 5])
# layer(tf.zeros([10, 5]))
# print(layer.variables)
# print(layer.trainable_variables)

model.summary()

# 关注 mean_squared_error 会发现与 自定义loss值相同，证明替换成功
model.compile(loss=costomized_mse, optimizer="sgd", metrics=["mean_squared_error"])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

model.evaluate(x_test_scaled, y_test)
