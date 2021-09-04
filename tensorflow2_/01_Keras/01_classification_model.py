import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
'''
    分类模型 => Flatten * 1 + 全连接层 * 3  
    Flatten => 将多维向量 一维化
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

def show_single_image(img):
    plt.imshow(img, cmap="binary")
    plt.show()


# show_single_image(x_train[0])

def show_imgs(n_rows, n_cols, x_data, xlabel_data, class_names):
    assert len(x_data) == len(xlabel_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[xlabel_data[index]])
    plt.show()


# show_imgs(3, 5, x_train, xlabel_train, class_names)

# 激活函数准确率 relu 10%  sigmoid 80% ?
model = keras.models.Sequential();
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # 展平
model.add(keras.layers.Dense(300, activation="sigmoid"))  # 全连接层单元数 300
model.add(keras.layers.Dense(100, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="softmax"))

#第二种写法
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
                                 keras.layers.Dense(500, activation="sigmoid"),
                                 keras.layers.Dense(300, activation="sigmoid"),
                                 keras.layers.Dense(10, activation="softmax")]);

# relu: y = max(0, x)
# softmax: 将向量变成概率分布  x = [x1, x2, x3]
#          y=[e^x1/sum, e^x2/sum, e^x3/sum], sum = e^x1 + e^x2 + e^x3

# 交叉熵作为损失函数
# because y->one-hot->[],if y is number not [],use loss="sparse_categorical_crossentropy"
#                        if y is [] -> categorical_crossentropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",  # 优化器
              metrics=['accuracy'])  # 其他关注指标

for layer in model.layers:
    print(layer)
model.summary()

history = model.fit(x_train, xlabel_train, epochs=10, validation_data=(x_valid, xlabel_valid))

print(history.history)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)
