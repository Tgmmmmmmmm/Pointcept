#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v2
# @Time          : 2025-07-29 00:09:05
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : ​向量化位运算, 批量计算空间填充曲线编码 142.36 ms
# @version       : 2.0
"""
import numpy as np


def space_filling_encode_v2(
    coords: np.ndarray, M: np.ndarray, dim: int = 3, depth: int = 16
) -> np.ndarray:
    N = coords.shape[0]
    total_bits = dim * depth
    # 如果输入是PyTorch Tensor，转换为numpy数组
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    # 显式转换为整数类型
    coords = coords.astype(np.int64)

    # 为每个坐标生成二进制表示 (N x dim x depth)
    coord_bits = np.zeros((N, dim, depth), dtype=np.uint8)  # 使用更小的数据类型
    for d in range(dim):
        for b in range(depth):
            # 显式转换中间结果
            shifted = np.right_shift(coords[:, d], (depth - 1 - b))
            coord_bits[:, d, b] = np.bitwise_and(shifted, 1)

    # 按照z,y,x顺序拼接 (N x total_bits)
    v = coord_bits.transpose(0, 2, 1).reshape(N, -1)

    # 矩阵运算
    encodings = np.mod(v @ M.T, 2)

    # 生成最终编码（显式指定为无符号64位整数）
    powers = 1 << np.arange(total_bits - 1, -1, -1, dtype=np.uint64)
    codes = np.dot(encodings.astype(np.uint64), powers)

    return codes.astype(np.uint64)
