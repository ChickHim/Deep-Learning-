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
    利用 sklearn 实现超参数搜索
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

"""1.转化为 sklearn 的 model"""


def build_model(hidden_layers=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation="sigmoid",
                                 input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size, activation="sigmoid", ))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


# keras.wrappers.scikit_learn.KerasClassifier -> 分类
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]
history = sklearn_model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                            epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

"""
    2.定义参数集合 
        learning_rate 按照此分布取值
        reciprocal(x) = 1/(x*log(b/a))   a <= x <=b
"""
from scipy.stats import reciprocal

param_distribution = {
    "hidden_layers": [1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2)
}

"""3.利用  搜索参数"""
from sklearn.model_selection import RandomizedSearchCV

random_search_cv = RandomizedSearchCV(sklearn_model, param_distribution,
                                      n_iter=10,  # 从 param_distribution中取多少组参数集合
                                      n_jobs=1,
                                      cv=3)
# 会使用 cross_validation机制 : 训练分成 n份  n-1 训练  最后一份验证 默认 cv=3
random_search_cv.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                     callbacks=callbacks)

print(random_search_cv.best_estimator_)
print(random_search_cv.best_params_)
print(random_search_cv.best_score_)

model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled, y_test)
