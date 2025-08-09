#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v1.py
# @Time          : 2025-07-29 00:08:02
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 原始暴力版本空间填充曲线编码实现

# 功能描述：
# 本模块实现基于满秩矩阵的空间填充曲线编码算法。
# 该版本为原始实现，直接使用矩阵乘法计算，单次计算耗时约 1424.02 毫秒(size:3x16)。

# 算法原理：
# 1. 将每个坐标分量转换为二进制表示
# 2. 拼接为向量(坐标维度降低顺序)
# 3. 与满秩矩阵M进行模2矩阵乘法
# 4. 将结果二进制向量转换为uint64编码
"""

import numpy as np
from typing import List, Union


def int_to_bits(n: int, bits: int = 16) -> List[int]:
    """
    将整数转换为指定位数的二进制列表（高位在前）

    参数:
        n: 要转换的整数
        bits: 输出位数，默认为16

    返回:
        包含二进制位的列表，长度等于bits参数

    示例:
        >>> int_to_bits(5, 4)
        [0, 1, 0, 1]
    """
    return [int(b) for b in format(n, f"0{bits}b")]


def space_filling_encode_v1(
    coords: np.ndarray, M: np.ndarray, dim: int = 3, depth: int = 16
) -> np.ndarray:

    N = coords.shape[0]
    total_bits = dim * depth

    # 预分配二进制矩阵（避免动态扩展列表）
    binary_matrix = np.zeros((N, total_bits), dtype=int)

    # 将坐标转为二进制并填充矩阵
    for i in range(N):
        for j in range(dim):
            # 从最高维度到最低维度（z→y→x）
            coord_val = coords[i, dim - 1 - j]
            # 直接提取二进制位并填入矩阵
            binary_matrix[i, j * depth : (j + 1) * depth] = [
                (coord_val >> (depth - 1 - k)) & 1 for k in range(depth)
            ]

    # 矩阵乘法 + 模2运算
    encodings = np.mod(binary_matrix @ M.T, 2)

    # 使用位运算生成最终编码（替代手动循环）
    codes = np.zeros(N, dtype=np.uint64)
    for i in range(N):
        code = 0
        for bit in encodings[i]:
            code = (code << 1) | bit
        codes[i] = code

    return codes
