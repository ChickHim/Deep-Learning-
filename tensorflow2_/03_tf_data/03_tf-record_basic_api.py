import os, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
    tf-record: 是 tf 独有的一种文件格式
        其内容为 tf.train.Example
        Example 的内容为 Features -> {"key":tf.train.Feature}
        Feature 的内容是 tf.train.ByteList/ FloatList/ Int64List
        
"""

"""Feature"""
favourite_books = [name.encode('utf-8') for name in ["machine learning", "cc150"]]

favourite_books_bytelist = tf.train.BytesList(value=favourite_books)
print(favourite_books_bytelist)

hours_floatlist = tf.train.FloatList(value=[15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)

age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

"""Features"""
features = tf.train.Features(
    feature={
        "favourite_books": tf.train.Feature(bytes_list=favourite_books_bytelist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list),
    }
)
print("Features:", features)

"""Example"""
example = tf.train.Example(features=features)
print("Example:", example)

serialized_example = example.SerializeToString()
print(serialized_example)

"""tf-record 读取储存"""
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filename = "test.tfrecord"
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)

expected_features = {
    "favourite_books": tf.io.VarLenFeature(dtype=tf.string),  # 不定长元素
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([], dtype=tf.int64),  # 定长元素
}
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)

    books = tf.sparse.to_dense(example["favourite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
    print(example)

""" tf-record 压缩文件 储存和读取 """
filename_fullpath_zip = filename_fullpath + ".zip"
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)

    books = tf.sparse.to_dense(example["favourite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))
