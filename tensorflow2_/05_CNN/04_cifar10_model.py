import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image

"""
    通过 keras 处理图片数据 实现 数据增强
"""

"""
    访问其他文件夹的数据 会报错:
        UserWarning: Found 5000 invalid image filename(s) in x_col="filepath". These filename(s) will be ignored.
        .format(n_invalid, x_col)
        Found 0 validated image filenames belonging to 10 classes.
        
        [解决方案] 
            flow_from_dataframe 的 directory参数
            如果 x_col 包含指向每个图像文件 可以用 None
            用 None 替换了 别的文件夹名 后会好用
"""
data_dir = '../input/cifar-10/'
# data_dir = './cifar-10/'

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

train_folder = data_dir + 'train/'
test_folder = data_dir + 'test/'
train_labels_file = data_dir + 'trainLabels.csv'
test_csv_file = data_dir + 'sampleSubmission.csv'


def parse_csv_file(filepath, folder):
    """parsers csv files into(filename(path),label) format"""
    results = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        image_id, label_str = line.strip('\n').split(',')
        image_full_path = folder + image_id + '.png'
        results.append((image_full_path, label_str))
    return results


train_labels_info = parse_csv_file(train_labels_file, train_folder)
test_csv_info = parse_csv_file(test_csv_file, test_folder)

""" 查看数据集 """
import pprint

pprint.pprint(train_labels_info[0:5])
pprint.pprint(test_csv_info[0:5])
print(len(train_labels_info), len(test_csv_info))

"""转成 dataframe"""
train_df = pd.DataFrame(train_labels_info[0:45000], columns=['filepath', 'class'])
valid_df = pd.DataFrame(train_labels_info[45000:], columns=['filepath', 'class'])
test_df = pd.DataFrame(test_csv_info, columns=['filepath', 'class'])

print(train_df.head())
print(valid_df.head())
print(test_df.head())

""" 图片大小 """
height = 32
width = 32
channels = 3
batch_size = 32
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # 每个像素点 乘 1/255
    rotation_range=40,  # 图片随机旋转角度  -40到40之间
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # 剪切强度
    zoom_range=0.2,  # 缩放强度
    horizontal_flip=True,  # 是否随即反转
    fill_mode='nearest'  # 当你为图片做处理的时 填充像素的规则
)
"""
    directory 参数
    -> 指向包含所有图像的目录的路径
    -> 如果 x_col 包含指向每个图像文件 可以用 None

    class_mode 参数 
    -> "categorical", "binary", "sparse", "input", "other" or None 之一。 默认："categorical"。决定返回标签数组的类型：
            "categorical" 将是 2D one-hot 编码标签，好像多分类标签用这个，
            "binary" 将是 1D 二进制标签，
            "sparse" 将是 1D 整数标签，
            "input" 将是与输入图像相同的图像（主要用于与自动编码器一起使用），
            "other" 将是 y_col 数据的 numpy 数组，
            None, 不返回任何标签（生成器只会产生批量的图像数据，这对使用 model.predict_generator(), model.evaluate_generator() 等很有用）
"""
directory = None  # 其他文件夹的数据集
# directory = './'  # 本文件夹的数据集

train_generator = train_datagen.flow_from_dataframe(train_df,  # dataframe
                                                    directory=directory,  # 包含在 dataframe 中映射的所有图像
                                                    x_col='filepath',  # 包含目标图像文件夹的目录的列
                                                    y_col='class',  # dataframe 中将作为目标数据的列名，标签那一列
                                                    classes=class_names,
                                                    target_size=(height, width),  # 图片的大小
                                                    batch_size=batch_size,  # 多少图片为一组
                                                    seed=7,
                                                    shuffle=True,  # 混牌
                                                    class_mode='sparse',  # 控制 label 格式
                                                    )

"""
    验证集不需要过多的处理
    只需要缩放即可
"""
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_datagen.flow_from_dataframe(valid_df,
                                                    directory=directory,
                                                    x_col='filepath',
                                                    y_col='class',
                                                    classes=class_names,

                                                    target_size=(height, width),
                                                    batch_size=batch_size,
                                                    seed=7,
                                                    shuffle=False,
                                                    class_mode='sparse')

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
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
                        input_shape=[width, height, channels]),
    # 每个卷积层后面都是用 BN 会加快训练速度
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
epochs = 20

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

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = valid_datagen.flow_from_dataframe(test_df,
                                                   directory=directory,
                                                   x_col='filepath',
                                                   y_col='class',
                                                   classes=class_names,

                                                   target_size=(height, width),
                                                   batch_size=batch_size,
                                                   seed=7,
                                                   shuffle=False,
                                                   class_mode='sparse')

print(test_generator.samples)

test_predict = model.predict_generator(test_generator,
                                       workers=10,  # 并行数
                                       use_multiprocessing=False,  # 并行时 True-> 进程  False-> 线程
                                       )

"""查看数据集"""
print(test_predict.shape)

"""获取到 十个类的概率分布"""
print(test_predict[0:5])

"""获取 对应数据最大值的 索引"""
test_predict_class_indices = np.argmax(test_predict, axis=1)
print(test_predict_class_indices[0:5])

"""从索引中 获取对应的类名"""
test_predict_class = [class_names[index] for index in test_predict_class_indices]
print(test_predict_class[0:5])


def generate_submission(filename, predict_class):
    with open(filename, 'w') as f:
        f.write('id,label\n')
        for i in range(len(predict_class)):
            f.write('%d,%s\n' % (i + 1, predict_class[i]))


out_put = data_dir + 'sampleSubmission.csv'
generate_submission(out_put, test_predict_class)
