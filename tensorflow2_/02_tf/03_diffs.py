import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    自定义求导
        -> 01_keras.19_ is example
"""


def f(x):
    return 3. * x ** 2 + 2. * x - 1


def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)


print(approximate_derivative(f, 1.))


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2


print(approximate_gradient(g, 2., 3.))

"""上述过程使用 tf"""
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

"""使用 tf.GradientTape() 进行导数的求解"""
with tf.GradientTape() as tape:
    z = g(x1, x2)
print("tf - x1' :", tape.gradient(z, x1))

"""使用一次之后对象被释放 再次使用 error"""
try:
    print(tape.gradient(z, x2))
except RuntimeError as ex:
    print(ex)

"""一次性求出两个偏导"""
with tf.GradientTape() as tape:
    z = g(x1, x2)
print("tf - x1,x2' :", tape.gradient(z, [x1, x2]))
del tape

"""常量不能直接求导 需要 tape.watch()"""
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch([x1, x2])  # 不加的结果:  tf.constant - x1x2' : [None, None]
    z = g(x1, x2)
print("tf.constant - x1x2' :", tape.gradient(z, [x1, x2]))

"""对多个目标函数求导 结果会加起来"""
x = tf.Variable(5.)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
print("对多个目标函数求导:", tape.gradient([z1, z2], x))

"""二阶导"""
x1 = tf.Variable(3.)
x2 = tf.Variable(4.)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2]) for inner_grad in inner_grads]
print(outer_grads)
del inner_tape, outer_tape

"""自定义梯度下降"""
learning_rate = 0.1
x = tf.Variable(0.)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dx)  # x-=..
print("梯度下降:", x)

"""使用 SGD 代替 自定义梯度下降"""
learning_rate = 0.1
x = tf.Variable(0.)
optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dx, x), ])  # x.assign_sub(learning_rate * dx)
print("SGD:", x)
