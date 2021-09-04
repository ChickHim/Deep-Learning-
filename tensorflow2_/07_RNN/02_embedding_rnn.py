import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    分别通过
        单层单向 RNN  accuracy: 0.5099 -> 比猜强一点
        双向多层 RNN  accuracy: 0.6800 -> 模型过于复杂 过拟合严重
        双向单层 RNN  accuracy: 0.7704 -> 略微简化模型 效果变好
        
    实现 文本分类
        二分类 随机概率 0.5
"""
imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocab_size,
    index_from=index_from
)

word_index = imdb.get_word_index()
print('word_index len():', len(word_index))
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
    定义模型  单层 RNN
        keras.layers.SimpleRNN():
            1. 定义矩阵 [vocab_size, embedding_dim]
            2. 每一个样本 [1,2,3,4] -> [max_length, embedding_dim]
            3. 经过训练 -> [batch_size, max_length, embedding_dim]
            
        keras.layers.Bidirectional(SimpleRNN())
            封装之后变成双向 RNN
            
        keras.layers.GlobalAveragePooling1D():
            1. 消除 max_length 维度
                [batch_size, max_length, embedding_dim] -> [batch_size, embedding_dim]
"""

""" 单向单层 rnn """
single_rnn_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # 添加一个 rnn 层次
    keras.layers.SimpleRNN(
        units=64,  # 输出空间的维度
        return_sequences=False,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
    ),
    # 由于取得最后一步输出
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
], name='single_rnn_model')
single_rnn_model.summary()
single_rnn_model.compile(optimizer='adam', loss='binary_crossentropy',
                         metrics=['accuracy'])
history_single_rnn = single_rnn_model.fit(
    train_data, train_labels,
    epochs=30,
    batch_size=batch_size,  # 由于数据没有分 batch, 所以训练的时候分
    validation_split=0.2,  # 从训练集中划分 20% 作为验证集
)

""" 双向多层 rnn """
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # 双向 RNN
    keras.layers.Bidirectional(
        keras.layers.SimpleRNN(
            units=64,  # 输出空间的维度
            return_sequences=True,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
        )),
    keras.layers.Bidirectional(
        keras.layers.SimpleRNN(
            units=64,  # 输出空间的维度
            return_sequences=False,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
        )),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
], name='model')
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data, train_labels,
    epochs=30,
    batch_size=batch_size,  # 由于数据没有分 batch, 所以训练的时候分
    validation_split=0.2,  # 从训练集中划分 20% 作为验证集
)

""" 双向单层 rnn """
bi_rnn_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # 双向 RNN
    keras.layers.Bidirectional(
        keras.layers.SimpleRNN(
            units=32,  # 输出空间的维度
            return_sequences=False,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
        )),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
], name='bi_rnn_model')
bi_rnn_model.summary()
bi_rnn_model.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])

bi_rnn_history = bi_rnn_model.fit(
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
plot_learning_curves(history_single_rnn, 'accuracy', 30, 0, 1)
plot_learning_curves(history_single_rnn, 'loss', 30, 0, 2.5)

plot_learning_curves(history, 'accuracy', 30, 0, 1)
plot_learning_curves(history, 'loss', 30, 0, 2.5)

plot_learning_curves(bi_rnn_history, 'accuracy', 30, 0, 1)
plot_learning_curves(bi_rnn_history, 'loss', 30, 0, 2.5)

print("单向单层 rnn")
single_rnn_model.evaluate(
    test_data, test_labels,
    batch_size=batch_size
)

print("双向多层 rnn")
model.evaluate(
    test_data, test_labels,
    batch_size=batch_size
)

print("双向单层 rnn")
bi_rnn_model.evaluate(
    test_data, test_labels,
    batch_size=batch_size
)
