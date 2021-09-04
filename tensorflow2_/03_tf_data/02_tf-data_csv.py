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
    生成读取 csv文件
"""
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split

# test_size=0.25
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_valid_scaled = scaler.transform(x_valid)

output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def save_to_csv(output_dir, data, name_prefix, header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []

    for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        # print(file_idx, row_indices)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join([repr(col) for col in data[row_index]]))
                f.write("\n")

    return filenames


train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MidianHouseValue']
# print(header_cols)
# MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MidianHouseValue
# print(header_str)

train_filenames = save_to_csv(output_dir, train_data, "train", header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, "valid", header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, "test", header_str, n_parts=10)

"""读取"""

n_readers = 5
filename_dataset = tf.data.Dataset.list_files(train_filenames)  # 将文件名生成一个 dataset
data_set = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),  # 专门读取文本文件的API 空过去一行
    cycle_length=n_readers,
)
for line in data_set.take(15):
    print(line.numpy())


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parse_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parse_fields[0:-1]) # 转为向量
    y = tf.stack(parse_fields[-1:])
    return x, y


def csv_reader_dataset(filenames, n_readers=5, batch_size=32,
                       n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()  # 数据重复多少份 空为 无限份
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)  # 洗牌
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)  # 分别对其中的各个元素操作
    dataset = dataset.batch(batch_size=batch_size)  # 每次取多少数据
    return dataset


import pprint

train_set = csv_reader_dataset(train_filenames, batch_size=3).take(2)
for x_batch, y_batch in train_set:
    print("x:")
    pprint.pprint(x_batch)
    print("x:")
    pprint.pprint(y_batch)

batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="sigmoid", input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(train_set, validation_data=(valid_set),
                    steps_per_epoch=11160 // batch_size,  # 数据是无限的,每个 epoch跑多少数据
                    validation_steps=3870 // batch_size,
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

model.evaluate(test_set, steps=5160 // batch_size)
