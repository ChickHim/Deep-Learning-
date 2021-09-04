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
    子类API 写 Wide & Deep 模型
"""
housing = fetch_california_housing()
print(housing.DESCR)
# data => 8个特征
print(housing.data.shape)
# target => 加利福尼亚地区的房屋中位价
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
    使用子类API
        继承 keras.models.Model
"""


class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型层次"""
        self.hidden1_layer = keras.layers.Dense(30, activation="relu")
        self.hidden2_layer = keras.layers.Dense(30, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, input):
        """完成模型正向计算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output


model = WideDeepModel()
model.build(input_shape=(None, 8))

model.summary()
model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 100)
model.evaluate(x_test_scaled, y_test)
