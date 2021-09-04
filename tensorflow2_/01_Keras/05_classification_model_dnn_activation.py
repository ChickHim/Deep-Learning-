import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
    更改激活函数
        relu:
            1.使用 ReLU 得到的 SGD 的收敛速度会比 sigmoid/tanh 快很多
            2.训练时很脆弱 如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都dead了。
        selu:
            自带批归一化功能的激活函数 运算比批归一化快 在较短的时间内达到较高水准 准确率更高
        tanh: [-1,1]  均值 0
            1.tanh在特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果。
        sigmoid: (0,1)
            1.在特征相差比较复杂或是相差不是特别大时效果比较好 适合二分类
            2.计算量大 收敛缓慢 反向传播容易造成梯度消失 
            
        softmax:
            1.模拟 max行为 是大的更大，适用于多分类
            
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

model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

logdir = r'.\dnn-selu-callbacks'
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
