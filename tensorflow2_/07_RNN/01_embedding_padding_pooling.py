import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    实现 合并 + padding
        缺点: 
            1. 信息丢失
                多个 embedding 合并
                pad会产生噪音、 无主次
            2. 无效计算过多 低效
                有太多的 padding
"""
imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3

"""
    num_words 限定 此表的个数
    index_from 此表从几开始算
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocab_size,
    index_from=index_from
)

print(train_data[0], train_labels[0])  # [1, 14, 22, 16, 43,...] 1
print(train_data.shape, train_labels.shape)  # (25000,) (25000,)  -> 25000个 维度不确定
print(len(train_data[0]), len(train_data[1]))  # 218 189  -> 变长

print(test_data.shape, test_labels.shape)  # (25000,) (25000,)

word_index = imdb.get_word_index()
print(len(word_index))
# print(word_index)

word_index = {k: (v + index_from) for k, v in word_index.items()}

""" 填充 word_index """
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<END>"] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(word_id, "<UNK>") for word_id in text_ids])


print(decode_review(train_data[0]))

"""
    设置最大长度
     -> 高于 max 被截断
     -> 低于 max 补全
"""
max_length = 500

"""数据补全"""
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,  # list of list
    value=word_index['<PAD>'],  # 补充的值
    padding='post',  # padding的顺序  post-> padding放在句子后面, pre-> 句子前面
    maxlen=max_length,
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,  # list of list
    value=word_index['<PAD>'],  # 补充的值
    padding='post',  # padding的顺序  post-> padding放在句子后面, pre-> 句子前面
    maxlen=max_length,
)

print(train_data[0])

""" 每个 embedding都定义为长度16的向量 """
embedding_dim = 16

batch_size = 128

"""
    定义模型
        keras.layers.Embedding():
            1. 定义矩阵 [vocab_size, embedding_dim]
            2. 每一个样本 [1,2,3,4] -> [max_length, embedding_dim]
            3. 经过训练 -> [batch_size, max_length, embedding_dim]
            
        keras.layers.GlobalAveragePooling1D():
            1. 消除 max_length 维度
                [batch_size, max_length, embedding_dim] -> [batch_size, embedding_dim]
"""
model = keras.models.Sequential([
    # embedding
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # 合并
    keras.layers.GlobalAveragePooling1D(),
    # 进入全连接层
    keras.layers.Dense(64, activation='relu'),
    # 因为是 二分类 -> sigmoid 是一个不错的选择
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data, train_labels,
    epochs=30,
    batch_size=batch_size,  # 由于数据没有分 batch, 所以训练的时候分
    validation_split=0.2,  # 从训练集中划分 20% 作为验证集
)


# 学习曲线
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


"""
    验证集曲线上升 -> 出现过拟合
"""
plot_learning_curves(history, 'accuracy', 30, 0, 1)
plot_learning_curves(history, 'loss', 30, 0, 2.5)

model.evaluate(
    test_data, test_labels,
    batch_size=batch_size
)
