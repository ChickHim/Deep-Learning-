import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    将 keras 中 自带的 ResNet50 中
        最后几层变成可训练的
        前面层的仍然不变
        
    不在 model 中 add  keras.applications.ResNet50()
    这会在 model 中变为一层
    
    resnet50 = keras.applications.ResNet50()
    这样 可以操作 预设 ResNet的 所有的层
    之后在加在别的 model 里面即可
"""
data_dir = r'..\input\10_Monkey_Species'

print(os.listdir(data_dir))

train_dir = os.path.join(data_dir, 'training')
valid_dir = os.path.join(data_dir, 'validation')
label_file = os.path.join(data_dir, 'monkey_labels.txt')

labels = pd.read_csv(label_file, header=0)

""" ResNet 处理的图像为 224 * 224 """
height = 224
width = 224
channels = 3
batch_size = 24
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    #  ResNet50 预处理图像的函数
    preprocessing_function=keras.applications.resnet50.preprocess_input,

    rotation_range=40,  # 图片随机旋转角度  -40到40之间
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True,  # 是否随即反转
    fill_mode='nearest'  # 当你为图片做处理的时 填充像素的规则
)
train_generator = train_datagen.flow_from_directory(train_dir,  # 文件路径
                                                    target_size=(height, width),  # 图片的大小
                                                    batch_size=batch_size,  # 多少图片为一组
                                                    seed=7,
                                                    shuffle=True,  # 混牌
                                                    class_mode='categorical',  # 控制 label 格式
                                                    )

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,
)
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(height, width),
                                                    batch_size=batch_size,
                                                    seed=7,
                                                    shuffle=False,
                                                    class_mode='categorical')

print(train_generator.samples, valid_generator.samples)

"""此处直接获取 预设的网络"""
resnet50 = keras.applications.ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
)

resnet50.summary()

for layer in resnet50.layers[0:-5]:
    layer.trainable = False

resnet50_new = keras.models.Sequential([
    resnet50,
    keras.layers.Dense(num_classes, activation="softmax"),
])

resnet50_new.summary()

resnet50_new.compile(loss='categorical_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])

epochs = 10
history = resnet50_new.fit_generator(train_generator,
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
