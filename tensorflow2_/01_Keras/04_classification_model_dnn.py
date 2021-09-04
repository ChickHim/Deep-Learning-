import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
    深度神经网络
        增加层数
    批归一化
        每一层的输入都做一次归一化
        缓解梯度消失
    
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

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))

""" 层数增加 => 深度神经网络 """
for _ in range(20):
    """ 批归一化放在激活函数之后 """
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())

    """ 批归一化放在激活函数之前 """
    # model.add(keras.layers.Dense(100))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

logdir = r'.\dnn-callbacks'
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
commonTools.plot_learning_curves(history, 3)
model.evaluate(x_test_scaled, xlabel_test)
