import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    分析数据集
"""

train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

"""原始数据集"""
print(train_df.head())
print(eval_df.head())

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

"""数据集 分为 特征 和 结果"""
print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())

""" pandas 统计"""
print(train_df.describe())

print(train_df.shape, eval_df.shape)

train_df.age.hist(bins=20)
plt.show()

train_df.sex.value_counts().plot(kind='barh')
plt.show()

train_df['class'].value_counts().plot(kind='bar')
plt.show()

survived_data = pd.concat([train_df,y_train],axis=1).groupby('sex').survived.mean()
print(survived_data)
survived_data.plot(kind='bar')
plt.show()
