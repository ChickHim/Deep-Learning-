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
    超参数
        训练之前设置 在训练中不会改变的参数
        
    超参数搜索
        在训练之前对超参数优化
            网格搜索 随机搜索 遗传算法 启发式搜索(AutoML)
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
    超参数搜索简单示意
        learning_rate = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
"""
learning_rate = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for lr in learning_rate:
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="sigmoid", input_shape=x_train.shape[1:]),
        keras.layers.Dense(1),
    ])
    myOptimizer = keras.optimizers.SGD(lr)
    model.compile(loss='mean_squared_error', optimizer=myOptimizer)
    callbacks = []  # [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

    history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                        epochs=100, callbacks=callbacks)
    histories.append(history)

import commonTools

for lr, history in zip(learning_rate, histories):
    print("learning_rate: "+lr)
    commonTools.plot_learning_curves(history, 1)

