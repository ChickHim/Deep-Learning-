# coding:utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
seed = 23455

# 基于 seed 产生随机数
rng = np.random.RandomState(seed)

# 随机数返回 32行 2列的矩阵 表示 32组数据 作为输入数据集
X = rng.rand(32, 2)

# 从 X取出一行 判断如果和小于1 给Y赋值 1  如果和不小于 1 给 Y赋值 0
# 作为输入数据集的标签
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print(X)
print(Y)

# 1 定义为神经网络的输入、参数和输出，定义前向传播过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 2 定义损失函数 及 反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 3 生成会话 训练 Steps轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前未经训练的参数取值
    print('w1', sess.run(w1))
    print('w2', sess.run(w2))

    # 训练模型
    STEPS = 3001
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print(i, 'steps', 'loss=', total_loss)

    print('w1', sess.run(w1))
    print('w2', sess.run(w2))
