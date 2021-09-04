import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    tf.data.Dataset()
"""

# 在内存中构建数据集
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))  # 参数可 numpy数组,字节流,list 等
print(dataset)

for item in dataset:
    print(item)

dataset = dataset.repeat(3).batch(7)  # 重复 3次, 每次选 7个
for item in dataset:
    print(item)

# interleave: 对 dataset中现有的元素进行处理
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),  # map_fn 如何处理
    cycle_length=5,  # cycle_length 并行处理元素个数
    block_length=5,  # block_length  每次取多少个元素
)
for item in dataset2:
    print(item)

# 输入元素变为元组
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)
for item_x, item_y in dataset3:
    print(item_x, item_y.numpy())

# 传入字典得字典
dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x, "label": y})
for item in dataset4:
    print(item)
    print(item["feature"].numpy(), item["label"].numpy())
