import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    metric -> loss
        mse: 均方差
"""
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))  # 9
print((metric([0.], [1.])))  # 5 -> [9 + (1 - 0)^2] / 2  叠加再平均
print(metric.result())  # 5

metric.reset_states()  # 重置清零
metric([1.], [3.])
print(metric.result())
