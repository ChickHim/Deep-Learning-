import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
    dropout
        随机弃用层内的神经元
        防止过拟合 只记样本 提高泛化
        若不存在过拟合现象 使用时会导致准确率下降
    
    实验结果 20层网络
            不加 dropout 准确率 0.873
            后5层 dropout 准确率 0.674
            最后一层 dropout 准确率 0.8742
            
    防止过拟合方式
        1. 正则化
        2. dropout
        3. 降低模型尺寸 (10层 减到 5层)
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

for _ in range(20):
    model.add(keras.layers.Dense(100, activation="selu"))

"""
    这里声明 dropout 会使得前一层变成 dropout
    rate: 丢掉的比例
        AlphaDropout: 在 均值方差 归一化性质 不变的情况下 dropout 
        Dropout: 普通的 dropout
"""
model.add(keras.layers.AlphaDropout(rate=0.5))
# model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

logdir = r'.\dnn-selu-dropout-callbacks'
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
