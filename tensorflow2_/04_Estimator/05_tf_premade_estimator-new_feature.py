import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    交叉特征 cross feature -> 对 两个离散特征 做 笛卡尔积
        
        cross feature:
            age -> [1, 2, 3, 4, 5]
            gender -> [male, female]
            
            age_x_gender -> [(1, male), (2, male), (3, male), (4, male), (5, male),
                             (1, female), (2, female), (3, female), (4, female), (5, female) ]
        
        作用: 能够更好的刻画数据
              如果 数据全是 离散数据 全部进行交叉特征处理 会导致 过拟合
"""

train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

categorical_columns = {"sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"}
numeric_columns = {"age", "fare"}
feature_columns = []

"""处理离散数据"""
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()  # .unique()返回有哪些不同的值

    """
        tf.feature_column.indicator_column()
            对离散特征列进行 one-hot 编码
            
        tf.feature_column.categorical_column_with_vocabulary_list(key, list)
            生成特征列
    """

    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column,  # 列名
                vocab)  # 词典
        ))

"""处理连续数据"""
for numeric_column in numeric_columns:
    """连续特征可以直接输入，所以直接封装"""
    feature_columns.append(tf.feature_column.numeric_column(numeric_column, dtype=tf.float32))

"""
    交叉特征
        ->  交叉特征会大幅度扩大特征的数量
            hash_bucket_size参数用于控制此数量
            
            例如 age_x_sex 出现 10000 个交叉特征 放入 100个 hash_bucket 中 -> 交叉特征仅增加了 100
            
            为什么可以减少 -> 交叉特征是非常稀疏的数据 部分位置可重用
            
    LinearClassifier 可用 crossed_columns 或 indicator_column
    DNNClassifier    只能使用 indicator_column
        -> 都封装成 indicator_column
             加到   feature_columns
"""

crossed_columns = tf.feature_column.crossed_column(['age', 'sex'],
                                                   hash_bucket_size=100)
feature_columns.append(tf.feature_column.indicator_column(crossed_columns))

"""数据集"""


def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


train_dataset = make_dataset(train_df, y_train, epochs=100)
eval_dataset = make_dataset(eval_df, y_eval, epochs=1, shuffle=False)

####################################################################################

baseline_output_dir = 'baseline_model_new_features'
if not os.path.exists(baseline_output_dir):
    os.mkdir(baseline_output_dir)

baseline_estimator = tf.compat.v1.estimator.BaselineClassifier(  # 用 v2 会报错
    model_dir=baseline_output_dir,  # 输入地址
    n_classes=2,  # 一共分多少类
)

baseline_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=95))

result = baseline_estimator.evaluate(
    input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))
print(result)

####################################################################################

linear_output_dir = 'linear_model_new_features'
if not os.path.exists(linear_output_dir):
    os.mkdir(linear_output_dir)

linear_estimator = tf.estimator.LinearClassifier(
    model_dir=linear_output_dir,  # 输入地址
    n_classes=2,  # 一共分多少类
    feature_columns=feature_columns,  # 使用 feature_columns 解析由 feature_columns 组成的 dataset
)

linear_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=95))

result = linear_estimator.evaluate(
    input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))
print(result)

####################################################################################

dnn_output_dir = 'dnn_model_new_features'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)

dnn_estimator = tf.estimator.DNNClassifier(
    model_dir=dnn_output_dir,  # 输入地址
    n_classes=2,  # 一共分多少类
    feature_columns=feature_columns,  # 使用 feature_columns 解析由 feature_columns 组成的 dataset
    hidden_units=[128, 128],  # 需要定义 神经网络层数 每层单元数 -> 2层 每层 128个单元
    activation_fn=tf.nn.relu,  # 激活函数
    optimizer=tf.optimizers.Adam,  # 优化器 , linear_estimator 也有，只是没用
)

dnn_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=95))

result = dnn_estimator.evaluate(
    input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))
print(result)
