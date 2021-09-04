# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

seed = 2  # 随机种子，保证随机数一致


def generateds():
    # 基于 seed产生随机数
    rdm = np.random.RandomState(seed=seed)
    X = rdm.randn(300, 2)
    # 从 X 这个 300行 2列的矩阵中取出一行  判断如果两个坐标的平方和小于 2  给 Y赋值 1  其余赋值 0
    # 作为输入数据集的标签 （正确答案）
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]

    # 遍历 Y中每个元素
    Y_c = [['red' if y else 'blue'] for y in Y_]

    # 对数据集 X和标签 Y进行 shape整理 第一个元素为 -1表示 n行 随第二个参数计算得到 第二个元素表示多少列 把 X整理成 n行 2列，Y n行1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c
