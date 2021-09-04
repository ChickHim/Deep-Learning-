import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing

"""
    使用 04_text_generation 中训练好的参数
    生成 文本生成模型
"""

input_filepath = '../input/shakespeare/shakespeare.txt'
text = open(input_filepath, 'r').read()
""" 1. 生成词表 """
vocab = sorted(set(text))

""" 2. 建立映射  char -> id """
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

""" 3. 数据 转为 id """
text_as_int = np.array([char2idx[c] for c in text])

output_dir = "./text_generation_checkpoints"

vocab_size = len(vocab)
embedding_dim = 256
rnn_unit = 1024

"""
    由于文件命名方式的原因 不可以引用 所以复制过来
"""


def build_model(vocab_size, embedding_dim, rnn_unit, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        keras.layers.SimpleRNN(rnn_unit, return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model


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

    for _ in range(num_generate):
        """
            model inference-. predictions
            predictions : [batch_size, input_eval_len, vocab_size]
        """
        predictions = model(input_eval)

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
