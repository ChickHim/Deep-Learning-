import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
  数据归一化：使得预处理的数据被限定在一定的范围内（比如[0,1]或者[-1,1]）
        目的: 消除奇异样本数据导致的不良影响
              加快训练速度
        原理: 均值为0 方差为1
            Min-max: x = (x - min)/(max-min)
            Z-score: x = (x - u)/ std 
                    u->均值   std->方差
'''
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Skirt', 'Sneaker', 'Bag', 'Ankle boot']

# 获取数据
fashion_mnist = keras.datasets.fashion_mnist
# x_ 图片   xlabel 图片对应的标题
(x_train_all, xlabel_train_all), (x_test, xlabel_test) = fashion_mnist.load_data()
# 训练集 验证集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
xlabel_valid, xlabel_train = xlabel_train_all[:5000], xlabel_train_all[5000:]
print('验证集', x_valid.shape, xlabel_valid.shape)
print('训练集', x_train.shape, xlabel_train.shape)
print('测试集', x_test.shape, xlabel_test.shape)
print('max:', np.max(x_train), ', min:', np.min(x_train))

# 归一化 Z-score
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
scaler = standard_scaler;
# 输入数据需要二维 => x_train : [None, 28, 28] -> [None, 784] 之后再变回来
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)

# fit_transform 会将均值方差记录下来，后面将会用训练集的均值方差
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)

print('max_scaled:', np.max(x_train_scaled), ', min_scaled:', np.min(x_train_scaled))

model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(500, activation="sigmoid"),
                                 keras.layers.Dense(300, activation="sigmoid"),
                                 keras.layers.Dense(10, activation="softmax")]);

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

history = model.fit(x_train_scaled, xlabel_train, epochs=10, validation_data=(x_valid_scaled, xlabel_valid))
print(history.history)

import commonTools
commonTools.plot_learning_curves(history, 1)

model.evaluate(x_test_scaled, xlabel_test)
