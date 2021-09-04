import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    文本生成
        数据处理
        1. 生成词表
        2. 建立映射  char -> id
        3. 数据 转为 id
        4. 文本生成模型的输入和输出: abcd -> bcd<eos>
            预测下一个字符的模型
            
        定义模型
"""

input_filepath = '../input/shakespeare/shakespeare.txt'
text = open(input_filepath, 'r').read()

print('text len(): ', len(text))
print(text[0:100])

""" 1. 生成词表 """
vocab = sorted(set(text))
print('vocab len(): ', len(vocab))
print(vocab)

""" 2. 建立映射  char -> id """
char2idx = {char: idx for idx, char in enumerate(vocab)}
print(char2idx)
idx2char = np.array(vocab)
print(idx2char)

""" 3. 数据 转为 id """
text_as_int = np.array([char2idx[c] for c in text])
print('text_as_int[0:10] : ', text_as_int[0:10])
print('text[0:10] : ', text[0:10])

""" 4. 文本生成模型的输入和输出 """


def split_input_target(id_text):
    """
    abcde -> abcd, bcde
    """
    return id_text[0:-1], id_text[1:]


char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1, drop_remainder=True)

for ch_id in char_dataset.take(2):
    print(ch_id, idx2char[ch_id])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(''.join(idx2char[seq_id.numpy()])))

seq_dataset = seq_dataset.map(split_input_target)
for item_input, item_output in seq_dataset.take(2):
    print('\nitem_input', item_input.numpy())
    print('\nitem_output', item_output.numpy())

batch_size = 64
buffer_size = 10000

seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

""" 建立模型 """

vocab_size = len(vocab)  # 类别数
embedding_dim = 256
rnn_unit = 1024

""" 不设置激活函数 可以当作 logits 传给随机采样 """


def build_model(vocab_size, embedding_dim, rnn_unit, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(
            units=rnn_unit,
            stateful=True, # 为初始化状态，初始化下一个索引i的批次样本
            recurrent_initializer='glorot_uniform', # 用来初始化内核权重矩阵，用于对输入进行线性转换
            return_sequences=True
        ),
        keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size, embedding_dim, rnn_unit, batch_size)
model.summary()

""" 获取概率分布 """
for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)  # -> (64, 100, 65)

"""
    随机采样
        基于上面的概率分布生成一段话
    贪心策略
        基于概率规划生成一句话
"""
"""
    logits=example_batch_predictions[0] -> 选择 (0, 100, 65) -> 数据维度 (100, 65)
    (100, 65) 对 100 这个维度的每个位置做 example -> (100, 1)
"""
sample_indices = tf.random.categorical(logits=example_batch_predictions[0], num_samples=1)
print(sample_indices)  # -> (100, 1) 的数据

""" (100, 1) -> (100, ) """
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)

""" 查看数据 """
print('Input: ', repr(''.join(idx2char[input_example_batch[0]])))
print()
print('Output: ', repr(''.join(idx2char[target_example_batch[0]])))
print()
print('Predictions: ', repr(''.join(idx2char[sample_indices])))

"""自定义 loss """


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)
example_loss = loss(target_example_batch, example_batch_predictions)
print(example_loss.shape)
print(example_loss.numpy().mean())

""" 保存训练过的 参数 """
output_dir = "./text_generation_checkpoints_lstm"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

checkpoint_prefix = os.path.join(output_dir, 'chpt{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

epochs = 100
history = model.fit(seq_dataset, epochs=epochs, callbacks=checkpoint_callback)

""" 最后的 model """
print(tf.train.latest_checkpoint(output_dir))

model = build_model(vocab_size, embedding_dim, rnn_unit, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(output_dir))
model.build(tf.TensorShape([1, None]))

"""
    根据 sequence A -> model -> 预测出 b
    A.append(b) -> B
    B -> model -> c
"""

model.summary()


def generate_text(model, start_string, num_generate=1000):
    input_eval = [char2idx[ch] for ch in start_string]

    """
        纬度扩展
            -> 在 input_eval 前面再加一个维度
    """
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    """
        temperature > 1  -> random
        temperature < 1  -> greedy
    """
    temperature = 0.5

    for _ in range(num_generate):
        """
            model inference-. predictions
            predictions : [batch_size, input_eval_len, vocab_size]
        """
        predictions = model(input_eval)

        """
            贪心算法或者随机算法 加强 计算过程
        """
        predictions = predictions / temperature
        """
            sample -> ch -> text_generated
            消除 第 0 维
            predictions : [input_eval_len, vocab_size]
            predictions_id: [input_eval_len, 1]
        """
        predictions = tf.squeeze(predictions, 0)
        predictions_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        text_generated.append(idx2char[predictions_id])

        """ update input_eval """
        input_eval = tf.expand_dims([predictions_id], 0)

    return start_string + ''.join(text_generated)


new_text = generate_text(model, 'All: ')
print(new_text)
