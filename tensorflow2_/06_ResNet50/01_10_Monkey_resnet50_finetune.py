import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    迁移学习
        从 keras 自带的 ResNet50模型迁移过来模型
    
    ResNet: 残差网络
        cnn 的一种变异
        
        ResNet50 -> 具有50个层次的残差网络
                    同理 还有 ResNet101
        证明 网络可以走向更深 
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

"""
    keras.applications.resnet50.preprocess_input
        -> ResNet50 预处理图像的函数，会实现归一化，所以可以去掉 rescale
"""
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
"""
    验证集也需要 ResNet50 预处理图像的函数
"""
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

for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    print(y)

resnet50_finetune = keras.models.Sequential()

"""
    keras.applications.ResNet50() 参数说明
    include_top: Ture / False   是否保留顶层的全连接网络
    weights:     None ->      随机初始化， 即不加载预训练权重 从头训练
                'imagenet’ -> 加载预训练权重， 下载训练好的模型 初始化我们自己建立的网络
    pooling:     当include_top=False时，该参数指定了池化方式
                 None -> 不池化，最后一个卷积层的输出为4D张量
                 'avg' -> 全局平均池化
                 'max' -> 全局最大值池化
    classes：    可选，图片分类的类别数，仅当include_top=True并且不加载预训练权重时可用。
    input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，
                 图片的宽高必须大于197，如(200,200,3)
                 默认为 224 * 224
"""
resnet50_finetune.add(keras.applications.ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',  # None -> 从开头是训练, imagenet -> 下载训练好的模型
))

""" 由于去掉了最后一层, 所以在此处加上 """
resnet50_finetune.add(keras.layers.Dense(num_classes, activation='softmax'))

""" 由于第一层加载训练好的模型，所以不需要再训练 """
resnet50_finetune.layers[0].trainable = False

resnet50_finetune.summary()

""" 在此处，sgd 会比 adam 好许多 """
resnet50_finetune.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

epochs = 10
history = resnet50_finetune.fit_generator(train_generator,
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
