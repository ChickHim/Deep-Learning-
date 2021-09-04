import os, sys, time, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
  回调函数
    1. TensorBoard: tensorboard --logdir='path'
    2. EarlyStopping: 提前结束
    3. ModelCheckpoint: 保存模型
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

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler();
# 输入数据需要二维 => x_train : [None, 28, 28] -> [None, 784] 之后再变回来
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28, 28)

model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(500, activation="sigmoid"),
                                 keras.layers.Dense(300, activation="sigmoid"),
                                 keras.layers.Dense(10, activation="softmax")])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

'''
    回调函数
    TensorBoard: 需要文件夹
    EarlyStopping: 
    ModelCheckpoint: 需要文件名
'''
logdir = ".\\callbacks"
if not os.path.exists(logdir):
    os.mkdir(logdir)
out_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    # 保存最好的模型，False => 保存最近的模型
    keras.callbacks.ModelCheckpoint(out_model_file, save_best_only=True),
    # monitor= 检测量
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]

history = model.fit(x_train_scaled, xlabel_train,
                    epochs=10, validation_data=(x_valid_scaled, xlabel_valid),
                    callbacks=callbacks)
print(history.history)

import commonTools
commonTools.plot_learning_curves(history, 1)
model.evaluate(x_test_scaled, xlabel_test)
