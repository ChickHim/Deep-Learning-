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
    自定义求导过程 代替 model.fit(...)
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

# batch 遍历训练集 metric
# 求导
# epoch结束 验证集 metric

epochs = 100
batch_size = 32
step_per_epoch = len(x_train_scaled) // batch_size  # 整除
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()

"""劣质随机遍历数据"""


def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="sigmoid", input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(step_per_epoch):  # 每 epoch 分批次训练
        x_batch, y_batch = random_batch(x_train_scaled, y_train, batch_size=batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)  # 获取预测值
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_pred, y_batch)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)  # 一一对应组合起来
        optimizer.apply_gradients(grads_and_vars)
        print("\rEpoch:", epoch, "  train mse:", metric.result().numpy(), end="")

    y_valid_pred = model(x_valid_scaled)
    vailed_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t", "vailed mse:", vailed_loss.numpy())

"""

model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = []  # [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid),
                    epochs=100, callbacks=callbacks)

"""
