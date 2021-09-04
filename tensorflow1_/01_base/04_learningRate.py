# coding:utf-8

import tensorflow as tf
import numpy as np

'''
    学习率大 震荡不收敛
    学习率小 收敛速度慢
    => 指数衰减学习率
'''

LEARNING_RATE_BASE = 0.1  # 学习率初始值
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 更新学习率的次数  => 总样本数/BATCH_SIZE

# 运行几轮 BATCH_SIZE 计数器，初始值 0 不被训练
global_step = tf.Variable(0, trainable=False)

# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                           staircase=True)

# 定义待优化参数 初值 5
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数 loss
loss = tf.square(w + 1)
# 反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)

# 会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After', i, 'steps: global step is', global_step_val, ', w is', w_val, ', learning rate is',
              learning_rate_val, ', loss is', loss_val)
