import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

'''
  tf1.0 静态图
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
print('max:', np.max(x_train), ', min:', np.min(x_train))

# 归一化 Z-score
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
scaler = standard_scaler;
# 输入数据需要二维 => x_train : [None, 28, 28] -> [None, 784] 之后再变回来
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28 * 28)

# fit_transform 会将均值方差记录下来，后面将会用训练集的均值方差
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28 * 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)
).reshape(-1, 28 * 28)

# 更改数据类型
xlabel_train = np.asarray(xlabel_train, dtype=np.int64)
xlabel_valid = np.asarray(xlabel_valid, dtype=np.int64)
xlabel_test = np.asarray(xlabel_test, dtype=np.int64)

print('max_scaled:', np.max(x_train_scaled), ', min_scaled:', np.min(x_train_scaled))

"""创建 dataset"""


def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


batch_size = 20
epochs = 10
dataset = make_dataset(x_train_scaled, xlabel_train, epochs=epochs, batch_size=batch_size)

"""
    tf 1.x 中 必须使用 dataset.make_one_shot_iterator() 和 make_initializable_iterator() 的 .get_next() 才能获取下一组数据
        dataset_iterator
            1. 自动初始化
            2. 不能被重新初始化
"""
dataset_iter = dataset.make_one_shot_iterator()
x, y = dataset_iter.get_next()
with tf.Session() as sess:
    x_val, y_val = sess.run([x, y])
    print(x_val.shape)
    print(y_val.shape)

"""
    静态图
        placeholder 占位符 -> 先构建图 再向占位符 传数据
"""

hidden_units = [100, 100]
class_num = 10

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation=tf.nn.relu)

logits = tf.layers.dense(input_for_next_layer, class_num)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

"""查看占位符"""
print(x)
print(logits)

init = tf.global_variables_initializer()
train_steps_per_epoch = x_train_scaled.shape[0] // batch_size

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            """
                不需要再使用 feed_dict={} 传值
                自动调用 dataset_iterator
            """
            loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op])
            print('\r[Train] epoch: %d, step: %d, loss: %3.5f, accuracy: %2.2f' % (epoch, step, loss_val, accuracy_val),
                  end="")
