# coding:utf-8

import tensorflow as tf
import numpy as np

"""
    损失函数大致分为三种
        1.mse Mean Squared Error
        2.自定义
        3.ce Cross Entropy 表征两个 概率分布 之间的距离
"""

PROFIT = 1
COST = 9
BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 定义输入、参数、输出、前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 自定义 损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST * (y - y_), PROFIT * (y_ - y)))


# ce
ce = tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
# 可用下替换上
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)


train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 3 生成会话 训练 Steps轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数取值
    print('w1', sess.run(w1))

    # 训练模型
    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        print(start)
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print(i, 'steps', 'loss=', total_loss)

    print('w1', sess.run(w1))
