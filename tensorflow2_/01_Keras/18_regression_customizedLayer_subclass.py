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
    利用 子类的方式实现 自定义 Dense Layer
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

"""自定义 dense layer"""


class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 构建所需要的参数
        self.kernel = self.add_weight(name="kernel", shape=[input_shape[1], self.units],
                                      initializer="uniform", trainable=True)
        self.bias = self.add_weight(name="bias", shape=[self.units, ],
                                    initializer="zeros", trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, inputs):
        # 正向计算
        return self.activation(inputs @ self.kernel + self.bias)


model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation="sigmoid", input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
])
model.summary()
model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = []  # [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

model.evaluate(x_test_scaled, y_test)
