# coding:utf-8

import tensorflow as tf

'''
    生成随机数 (正态分布[2*3 矩阵], 标准差=2, 均值=0, 随机种子=1)
        随机种子去掉，每次生成的随机数相同
        tf.random_normal()      正态分布
        tf.truncated_normal()   去掉过大偏离点的正态分布
        tf.random_uniform()     平均分布
'''

w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
print(w)

'''
    tf.zeros([3,2])     全0数组
    tf.ones([3,2])      全1数组
    tf.fill([3,2],6)    全定值数组 全是6
    tf.constant()       直接在里面给值
'''

print('=' * 15, 'start', '=' * 15)

'''
    简单的两层神经网络
'''
# 定义输入和参数

# x = tf.constant([[0.7, 0.5]])
# shape 行数为 None 可以在 run() 时 feed 多组数据
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
