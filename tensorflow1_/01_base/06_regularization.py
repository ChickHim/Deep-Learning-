# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
    正则化
        在损失函数中引入模型复杂度指标
        利用给 w 加权值
        弱化训练数据的噪声

    公式
        loss = loss (y and y_) + Regularizer * loss(w)

            loss (y and y_) 模型中所有参数的损失函数
            Regularizer 用超参数 Regularizer 给出参数 w 在总 loss 中的比例，即 正则化的权重
            loss(w) w 是需要正则化的参数
'''

'''
    参数多需要正则化
    
    loss(w)
        L1：w 的绝对值求和
        L2：w 平方求和
'''

BATCH_SIZE = 30
seed = 2
# 基于 seed 产生随机数
rdm = np.random.RandomState(seed)
# 随机数返回 300行 2列的矩阵  表示 300组坐标点 (x0,x1)作为输入数据集
X = rdm.randn(300, 2)

# 从 X 这个 300行 2列的矩阵中取出一行  判断如果两个坐标的平方和小于 2  给 Y赋值 1  其余赋值 0
# 作为输入数据集的标签 （正确答案）
Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]

# 遍历 Y中每个元素
Y_c = [['red' if y else 'blue'] for y in Y_]

# 对数据集 X和标签 Y进行 shape整理 第一个元素为 -1表示 n行 随第二个参数计算得到 第二个元素表示多少列 把 X整理成 n行 2列，Y n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print('X:', X)
print('Y_', Y_)
print('Y_c', Y_c)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


# 定义神经网络输入、参数、输出、前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2  # 输出层不激活

# 定义损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 定义反向传播方法 （不含正则化）
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print(i, 'steps', loss_mse_v)

    # 生成 x,y -3 到 3之间步长为 0.01 的二维网格坐标点
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]

    # 拉直 xx,yy 合并成 2列矩阵  得到一个网络坐标点集合
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 将网格坐标点喂入神经网络  probs 为输出
    probs = sess.run(y, feed_dict={x: grid})

    # 将 probs调整为 xx的样子
    probs = probs.reshape(xx.shape)

    print("w1:", sess.run(w1))
    print("b1:", sess.run(b1))
    print("w2:", sess.run(w2))
    print("b2:", sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()


# 定义反向传播方法 （）正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print(i, 'steps', loss_v)

    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)

    print("w1:", sess.run(w1))
    print("b1:", sess.run(b1))
    print("w2:", sess.run(w2))
    print("b2:", sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()