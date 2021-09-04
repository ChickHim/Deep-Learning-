import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    三种常量
        tf.constant()  tf.ragged.constant()
        tf.SparseTensor(...)
    一种变量
        tf.Variable()
"""
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t)
# 取第二列以后
print(t[:, 1:])
# 只去第二列
print(t[..., 1])

# ops
print("=" * 10, "ops")
print(t + 10)
print(tf.square(t))
# t × t 的转置
print(t @ tf.transpose(t))

# numpy conversion
print("=" * 10, "numpy conversion")
print(t.numpy())
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))

# Scalars -> 0 维
print("=" * 10, "Scalars")
t = tf.constant(2.87)
print(t.numpy())
print(t.shape)

# strings
print("=" * 10, "strings")
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF8_CHAR"))
print((tf.strings.unicode_decode(t, "UTF8")))

# string array
print("=" * 10, "string array")
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t, "UTF8"))  # RaggedTensor 不规则数据

# ragged tensor -> 不规则数据
print("=" * 10, "ragged tensor")
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
print(r)
print(r[1])
print(r[1:2])

# ops on ragged tensor
print("=" * 10, "ops on ragged tensor")
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis=0))

# axis 维度数量一致才能拼接
r3 = tf.ragged.constant([[81], [91, 92], []])
print(tf.concat([r2, r3], axis=1))

# 不完整数据转为 tensor 空缺位置用 0 补齐
print(r.to_tensor())

# sparse tensor -> 不规则数据
print("=" * 10, "sparse tensor")
# indices= 必须排好序 否则报错
# 乱序可用 : x = tf.sparse.reorder(x1)
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])

print(s)
print(tf.sparse.to_dense(s))

# ops on sparse tensor
print("=" * 10, "ops on sparse tensor")
print(s * 2)
# 不能执行加法
try:
    s3 = s + 1
except TypeError as ex:
    print(ex)

s4 = tf.constant([[1., 2.], [3., 4.], [5., 6.], [7., 9.]])
print(tf.sparse.sparse_dense_matmul(s, s4))

# Variables
print("=" * 10, "Variable")
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v.value())  # -> 转为 Tensor
print(v.numpy())  # -> 获取一个值

# 变量赋值  不能用 = 
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())
