import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    读取 使用 tf record 文件
        TFrecord 是TF 特有的数据格式 有许多优化
"""

source_dir = "./generate_csv/"


# print(os.listdir(source_dir))


def get_filename_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results


train_filenames = get_filename_by_prefix(source_dir, "train")
valid_filenames = get_filename_by_prefix(source_dir, "valid")
test_filenames = get_filename_by_prefix(source_dir, "test")

# import pprint
#
# pprint.pprint(train_filenames)
# pprint.pprint(valid_filenames)
# pprint.pprint(test_filenames)

""" copy from 02_tf-data_csv """


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parse_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parse_fields[0:-1])  # 转为向量
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


batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)

""" csv 变成 tf.train.Example and serialize """


def serialize_example(x, y):
    input_feature = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        feature={
            "input_feature": tf.train.Feature(float_list=input_feature),
            "label": tf.train.Feature(float_list=label),
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


""" 完整的函数用于整合上面的功能 """


def csv_dataset_to_tfrecords(base_filename, dataset, n_shards, steps_per_shard, compression_type=None):  # 压缩 GZIP
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    all_filenames = []

    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}'.format(base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(steps_per_shard):
                # 解开 batch
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(serialize_example(x_example, y_example))
        all_filenames.append(filename_fullpath)
    return all_filenames


n_shards = 20
train_steps_per_shard = 11610 // batch_size // n_shards
valid_steps_per_shard = 3880 // batch_size // n_shards
test_steps_per_shard = 5170 // batch_size // n_shards

output_dir = "generate_tfrecords"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_basename = os.path.join(output_dir, "train")
valid_basename = os.path.join(output_dir, "valid")
test_basename = os.path.join(output_dir, "test")

train_tfrecord_filenames = csv_dataset_to_tfrecords(train_basename, train_set, n_shards, train_steps_per_shard)
valid_tfrecord_filenames = csv_dataset_to_tfrecords(valid_basename, valid_set, n_shards, valid_steps_per_shard)
test_tfrecord_filenames = csv_dataset_to_tfrecords(test_basename, test_set, n_shards, test_steps_per_shard)

expected_features = {
    "input_feature": tf.io.FixedLenFeature([8], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32),
}


def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example, expected_features)
    return example["input_feature"], example["label"]


def tfrecord_reader_dataset(filenames, n_readers=5, batch_size=32,
                            n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()  # 数据重复多少份 空为 无限份
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type=None),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)  # 洗牌
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)  # 分别对其中的各个元素操作
    dataset = dataset.batch(batch_size=batch_size)  # 每次取多少数据
    return dataset


batch_size = 32
tfrecord_train_set = tfrecord_reader_dataset(train_tfrecord_filenames, batch_size=batch_size)
tfrecord_valid_set = tfrecord_reader_dataset(valid_tfrecord_filenames, batch_size=batch_size)
tfrecord_test_set = tfrecord_reader_dataset(test_tfrecord_filenames, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="sigmoid", input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss='mean_squared_error', optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4)]

history = model.fit(tfrecord_train_set, validation_data=(tfrecord_valid_set),
                    steps_per_epoch=11160 // batch_size,  # 数据是无限的,每个 epoch跑多少数据
                    validation_steps=3870 // batch_size,
                    epochs=100, callbacks=callbacks)

import commonTools

commonTools.plot_learning_curves(history, 1)

model.evaluate(tfrecord_test_set, steps=5160 // batch_size)
