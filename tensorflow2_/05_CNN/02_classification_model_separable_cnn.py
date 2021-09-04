import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
    深度可分离卷积     
'''
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Skirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, xlabel_train_all), (x_test, xlabel_test) = fashion_mnist.load_data()

x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
xlabel_valid, xlabel_train = xlabel_train_all[:5000], xlabel_train_all[5000:]
print('验证集', x_valid.shape, xlabel_valid.shape)
print('训练集', x_train.shape, xlabel_train.shape)
print('测试集', x_test.shape, xlabel_test.shape)

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler();
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)

model = keras.models.Sequential()

"""
    深度可分离卷积卷积层
        model.add(keras.layers.SeparableConv2D()
        
        -> 在输入层使用普通卷积
        -> 后续 使用 可分离卷积
            
"""
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',activation='selu', input_shape=(28, 28, 1)))
model.add(keras.layers.SeparableConv2D(filters=32, kernel_size=3, padding='same',activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

"""展平 连接全连接层"""
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="selu"))

"""输出"""
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

"""
    summary() 对比 深度可分离卷积 比 CNN 的参数少得多
"""
model.summary()

logdir = r'.\separable_cnn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
out_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(out_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]

history = model.fit(x_train_scaled, xlabel_train,
                    epochs=10, validation_data=(x_valid_scaled, xlabel_valid),
                    callbacks=callbacks)
print(history.history)

import commonTools

commonTools.plot_learning_curves(history, 3)
model.evaluate(x_test_scaled, xlabel_test)

"""
    cnn  activation='relu' -> 0.988
    cnn  activation='selu' -> 0.991 图像显示出的训练过程也更好
    separable cnn activation='selu' -> 0.9700
    
    用准确率换取 参数数量 和 计算量 的减少
"""
