import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    tf.function()  @tf.function:
        使用 python函数和代码块 转变成 tf中的图
            图结构
                -> 会被 tf 优化 尤其是使用 GPU 的时候
                -> 可以导入导出为 SaveModel  实现 断点续传
        
    tf.autograph:
        tf 依赖的机制
    get_concrete_function:
        add input signature -> SavedModel
"""


def scaled_elu(z, scale=1.0, alpha=1.0):
    # z > 0? scale * z : scale * alpha * tf.nn.elu(z)

    # 判断大于
    is_positive = tf.greater_equal(z, 0.0)
    # 三元表达式 tf.where()
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


# 标量 直接操作
print(scaled_elu(tf.constant(-3.)))
# 向量 分别操作
print(scaled_elu(tf.constant([-3., -2.5])))

# 转图
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))

# 判断两者之间的关系
print(scaled_elu_tf is scaled_elu)
print(scaled_elu_tf.python_function is scaled_elu)

# 比谁快

time_start = time.time()
scaled_elu(tf.random.normal((1000, 1000)))
time_end = time.time()
print(time_end - time_start)

time_start = time.time()
scaled_elu_tf(tf.random.normal((1000, 1000)))
time_end = time.time()
print(time_end - time_start)

"""使用注解的方式"""


@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increase = tf.constant(1.)
    for _ in range(n_iters):
        total += increase
        increase /= 2.0
    return total


print(converge_to_2(20))

"""普通函数转成 tf之后 查看内部代码"""


def display_tf(func):
    code = tf.autograph.to_code(func)
    print(code)


display_tf(scaled_elu)

"""
    tf.Variable() 变量不能在函数内部定义  需要在外面进行初始化
    tf.constant() 随意
"""

var = tf.Variable(0.)  # 复制到函数内部 体验 bug


@tf.function  # 不加注解不能使用 tf变量
def add_21():
    return var.assign_add(21.)  # +=


print(add_21())

"""输入数据类型限制"""


@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name="x")])
def cube(z):
    return tf.pow(z, 3)


try:
    print(cube(tf.constant([1., 2., 3., ])))
except ValueError as ex:
    print("ex: ", ex)
print(cube((tf.constant([1, 2, 3]))))

""""""

cube_func = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32)
)
print(cube_func)
print(cube_func is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func is cube.get_concrete_function(tf.constant([1, 2, 3], tf.int32)))

print("获取 Concrete Function 的图: ", cube_func.graph)
print("图中的操作: ", cube_func.graph.get_operations())

pow_op = cube_func.graph.get_operations()[2]
print(pow_op)
print("inputs: ", list(pow_op.inputs))
print("outputs: ", list(pow_op.outputs))

print(cube_func.graph.as_graph_def())
print(cube_func.graph.get_operation_by_name("x"))
print(cube_func.graph.get_tensor_by_name("x:0"))
