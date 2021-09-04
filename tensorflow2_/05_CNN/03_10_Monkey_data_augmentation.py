import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    通过 keras 处理图片数据 实现 数据增强
    
"""
data_dir = r'..\input\10_Monkey_Species'

print(os.listdir(data_dir))

train_dir = os.path.join(data_dir, 'training')
valid_dir = os.path.join(data_dir, 'validation')
label_file = os.path.join(data_dir, 'monkey_labels.txt')

labels = pd.read_csv(label_file, header=0)
# print(labels)

""" 图片大小 """
height = 128
width = 128
channels = 3
batch_size = 64
num_classes = 10

"""
    ImageDataGenerator
        -> 用于处理 读取的图片
"""
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,  # 每个像素点 乘 1/255
    rotation_range=40,  # 图片随机旋转角度  -40到40之间

    # 水平垂直方向位移 增加鲁棒性
    # 值在0到1之间 单位为 比例， 大于1 单位为 像素
    # 位移值是 0到 0.2之间 随机
    width_shift_range=0.2,
    height_shift_range=0.2,

    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True,  # 是否随即反转
    fill_mode='nearest'  # 当你为图片做处理的时 填充像素的规则
)
"""
    利用 ImageDataGenerator 获取数据
"""
train_generator = train_datagen.flow_from_directory(train_dir,  # 文件路径
                                                    target_size=(height, width),  # 图片的大小
                                                    batch_size=batch_size,  # 多少图片为一组
                                                    seed=7,
                                                    shuffle=True,  # 混牌
                                                    class_mode='categorical',  # 控制 label 格式
                                                    )
"""
    验证集不需要过多的处理
    只需要缩放即可
"""
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(height, width),
                                                    batch_size=batch_size,
                                                    seed=7,
                                                    shuffle=False,
                                                    class_mode='categorical')

print(train_generator.samples, valid_generator.samples)

"""
    从 generator 中取数据
    
    经实验 selu 在 cnn 中 不一定好用  可能会无法学习
"""

for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    print(y)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                        input_shape=[width, height, channels]),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 300

"""
    数据是由 generator 产生 -> model.fit_generator()
"""
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples // batch_size,
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.samples // batch_size)

print(history.history.keys())


# 学习曲线
def plot_learning_curves(hsitory, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = hsitory.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 2.5)
