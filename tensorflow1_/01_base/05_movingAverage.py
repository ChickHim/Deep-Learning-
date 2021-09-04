# coding:utf-8

import tensorflow as tf
import numpy as np

'''
    滑动平均(影子值) 记录每个参数一段时间内 过往值的平均，增加了模型的泛化性
    像是给参数加了一个影子，参数变化，影子缓慢追随

    公式  影子 = 衰减率 * 影子 * （1 - 衰减率） * 参数
'''

w1 = tf.Variable(0, dtype=tf.float32)

# 运行几轮 BATCH_SIZE 计数器，初始值 0 不被训练
global_step = tf.Variable(0, trainable=False)
# 学习率衰减率
MOVING_AVERAGE_DECAY = 0.99

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# ema_op = ema.apply([w1]) 更新 w1
# tf.trainable_variables() 可以自动将所有的带训练的参数汇总为列表
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1, ema.average(w1)]))

    # 参数 w1 的值赋 1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新 step 和 w1 的值，模拟出 100 轮迭代后， w1 变为 10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 每次 sess.run() 更新一次 w1 的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))