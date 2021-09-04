import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pprint

from sklearn.datasets import fetch_california_housing

"""
     多输入 与 多输出
"""

housing = fetch_california_housing()
print(housing.DESCR)
# data => 8个特征  (20640, 8)
print(housing.data.shape)
# target => 加利福尼亚地区的房屋中位价  (20640,)
print(housing.target.shape)
pprint.pprint(housing.data[0:5])
pprint.pprint(housing.target[0:5])

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

"""
    多输入
"""
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
"""假设有需求在 hidden2 输出"""
output2 = keras.layers.Dense(1)(hidden2)

model = keras.models.Model(inputs=[input_wide, input_deep], outputs=[output, output2])

model.summary()
model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

"""拆分输入"""
x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

"""按照定义 传入变量"""
history = model.fit([x_train_scaled_wide, x_train_scaled_deep], [y_train, y_train],
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], [y_valid, y_valid]),
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

model.evaluate([x_test_scaled_wide, x_test_scaled_deep], [y_test, y_test])
