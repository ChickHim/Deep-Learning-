# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import generateds

'''
    反向传播
        训练网络、优化网络参数
'''

BATCH_SIZE = 30
STEPS = 40000
REGULARIZER = 0.01

LEARNING_RATE_BASE = 0.001  # 学习率初始值
LEARNING_RATE_DECAY = 0.999  # 学习率衰减率
LEARNING_RATE_STEP = 300 / BATCH_SIZE  # 更新学习率的次数  => 总样本数/BATCH_SIZE


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    '''
        生成随机数据
    '''
    X, Y_, Y_c = generateds.generateds()

    '''
        前向过程
    '''
    y = forward.forward(x, REGULARIZER)

    '''
        值数衰减学习率
    '''

    # 运行几轮 BATCH_SIZE 计数器，初始值 0 不被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义指数下降学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                               staircase=True)

    '''
        正则化的损失函数
    '''
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    '''
        训练过程
    '''
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE

            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print(i, 'steps', loss_v)

        # 生成 x,y -3 到 3之间步长为 0.01 的二维网格坐标点
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        # 拉直 xx,yy 合并成 2列矩阵  得到一个网络坐标点集合
        grid = np.c_[xx.ravel(), yy.ravel()]
        # 将网格坐标点喂入神经网络  probs 为输出
        probs = sess.run(y, feed_dict={x: grid})
        # 将 probs调整为 xx的样子
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__  == '__main__':
    backward()