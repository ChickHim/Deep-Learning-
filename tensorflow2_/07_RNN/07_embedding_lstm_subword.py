import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

"""
    分别通过
        subword - accuracy: 0.5002   单层单向 LSTM  accuracy: 0.5120 -> rnn accuracy: 0.5099
        subword - accuracy: 0.8393   双向多层 LSTM  accuracy: 0.8184 -> rnn accuracy: 0.6800
        subword - accuracy: 0.8267   双向单层 LSTM  accuracy: 0.7948 -> rnn accuracy: 0.7704
        
    实现 文本分类
        二分类 随机概率 0.5
"""

"""
    tensorflow_datasets 中有 subword 的数据集
    对原来的数据集进行替换
"""
dataset, info = tfds.load(
    'imdb_reviews/subwords8k',
    with_info=True,  # 是否携带信息
    as_supervised=True,  # 是否携带 label
)

train_dataset, test_dataset = dataset['train'], dataset['test']
print(info)

tokenizer = info.features['text'].encoder
print('vocabulary size {}'.format(tokenizer.vocab_size))

sample_string = 'Tensorflow is cool.'

tokenized_string = tokenizer.encode(sample_string)
print('tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('tokenized string is {}'.format(original_string))

for token in tokenized_string:
    print('{}--"{}"'.format(token, tokenizer.decode([token])))

buffer_size = 10000
batch_size = 64


train_dataset = train_dataset.shuffle(buffer_size)
"""根据 batch 中最长的样本 对本 batch 中的数据进行 padding"""
train_dataset = train_dataset.padded_batch(
    batch_size,
    tf.compat.v1.data.get_output_shapes(train_dataset)
)
test_dataset = test_dataset.padded_batch(
    batch_size,
    tf.compat.v1.data.get_output_shapes(test_dataset)
)

print(tf.compat.v1.data.get_output_shapes(train_dataset))
print(tf.compat.v1.data.get_output_shapes(test_dataset))

""" 每个 embedding都定义为长度16的向量 """
embedding_dim = 16
batch_size = 512
vocab_size = tokenizer.vocab_size
"""
    RNN 和 LSTM 参数相同

    定义模型  单层 RNN
        keras.layers.LSTM():
            1. 定义矩阵 [vocab_size, embedding_dim]
            2. 每一个样本 [1,2,3,4] -> [max_length, embedding_dim]
            3. 经过训练 -> [batch_size, max_length, embedding_dim]
            
        keras.layers.Bidirectional(LSTM())
            封装之后变成双向 LSTM
            
        keras.layers.GlobalAveragePooling1D():
            1. 消除 max_length 维度
                [batch_size, max_length, embedding_dim] -> [batch_size, embedding_dim]
"""

""" 单向单层 rnn """
single_lstm_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim),
    # 添加一个 rnn 层次
    keras.layers.LSTM(
        units=64,  # 输出空间的维度
        return_sequences=False,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
    ),
    # 由于取得最后一步输出
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
], name='single_lstm_model')
single_lstm_model.summary()
single_lstm_model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])
history_single_lstm = single_lstm_model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

""" 双向多层 rnn """
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim),
    # 双向 RNN
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=64,  # 输出空间的维度
            return_sequences=True,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
        )),
    keras.layers.Bidirectional(
        keras.layers.LSTM(
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
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

""" 双向单层 rnn """
bi_lstm_model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim),
    # 双向 RNN
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=32,  # 输出空间的维度
            return_sequences=False,  # 决定返回的结果是最后一步的输出(False) 还是完整序列(True)
        )),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
], name='bi_lstm_model')
bi_lstm_model.summary()
bi_lstm_model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

bi_lstm_history = bi_lstm_model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
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
plot_learning_curves(history_single_lstm, 'accuracy', 10, 0, 1)
plot_learning_curves(history_single_lstm, 'loss', 10, 0, 2.5)

plot_learning_curves(history, 'accuracy', 10, 0, 1)
plot_learning_curves(history, 'loss', 10, 0, 2.5)

plot_learning_curves(bi_lstm_history, 'accuracy', 10, 0, 1)
plot_learning_curves(bi_lstm_history, 'loss', 10, 0, 2.5)

print("单向单层 lstm")
single_lstm_model.evaluate(test_dataset)

print("双向多层 lstm")
model.evaluate(test_dataset)

print("双向单层 lstm")
bi_lstm_model.evaluate(test_dataset)
