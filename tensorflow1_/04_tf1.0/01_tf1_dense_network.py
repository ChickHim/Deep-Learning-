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

print('max_scaled:', np.max(x_train_scaled), ', min_scaled:', np.min(x_train_scaled))

"""
    静态图
        placeholder 占位符 -> 先构建图 再向占位符 传数据
"""

hidden_units = [100, 100]
class_num = 10

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.int64, [None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation=tf.nn.relu)

logits = tf.layers.dense(input_for_next_layer, class_num)

"""
    last_hidden_output * W = logits -> softmax -> prob
        1. logits -> softmax -> prob
        2. labels -> one_hot
        3. calculate cross entropy
"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)

"""取平均值之前先让 correct_prediction 变成 float """
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

"""用于训练网络，每使用一次，网络训练一次"""
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

"""查看占位符"""
print(x)
print(logits)

"""session"""
init = tf.global_variables_initializer()
batch_size = 20
epochs = 10
train_steps_per_epoch = x_train_scaled.shape[0] // batch_size
valid_steps = x_valid.shape[0] // batch_size

"""验证函数"""


def eval_with_session(sess, x, y, accuracy, images, labels, batch_size):
    eval_steps = images.shape[0] // batch_size
    eval_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step * batch_size:(step + 1) * batch_size]
        batch_label = labels[step * batch_size:(step + 1) * batch_size]
        accuracy_val = sess.run(accuracy,
                                feed_dict={
                                    x: batch_data,
                                    y: batch_label
                                })
        eval_accuracies.append(accuracy_val)
    return np.mean(eval_accuracies)


with tf.Session() as sess:
    # 运行初始化之后 图才会被构建调用
    sess.run(init)

    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train_scaled[step * batch_size:(step + 1) * batch_size]
            batch_label = xlabel_train[step * batch_size:(step + 1) * batch_size]
            """
                训练时可以训练多个算子 并按顺序获取值

                使用 feed_dict 填充数据给占位符
            """
            loss_val, accuracy_val, _ = sess.run(
                [loss, accuracy, train_op],
                feed_dict={
                    x: batch_data,
                    y: batch_label
                })
            print('\r[Train] epoch: %d, step: %d, loss: %3.5f, accuracy: %2.2f' % (epoch, step, loss_val, accuracy_val),
                  end="")
            valid_accuracy = eval_with_session(sess, x, y, accuracy, x_valid_scaled, xlabel_valid, batch_size)
            print('\t[Valid] acc: %2.2f' % valid_accuracy)
