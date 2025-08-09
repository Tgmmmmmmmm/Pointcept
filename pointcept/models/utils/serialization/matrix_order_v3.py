#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v3
# @Time          : 2025-07-30 09:16:04
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 进一步向量化 81.83 ms
"""
import numpy as np
from typing import Union


def space_filling_encode_v3(
    coords: np.ndarray, M: np.ndarray, dim: int = 3, depth: int = 16
) -> np.ndarray:
    N = coords.shape[0]
    total_bits = dim * depth
    # 如果输入是 PyTorch Tensor，转换为 NumPy 数组
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    # 向量化二进制转换
    shifts = np.arange(depth - 1, -1, -1)  # 预计算位移量
    coord_bits = ((coords[:, :, np.newaxis] >> shifts) & 1).astype(np.uint8)

    # 按z,y,x顺序拼接 (使用负步长翻转)
    v = coord_bits.transpose(0, 2, 1).reshape(N, -1)

    # 矩阵运算并确保结果为整数
    encodings = (v @ M.T) % 2

    # 生成最终编码（显式指定为无符号64位整数）
    powers = 1 << np.arange(total_bits - 1, -1, -1, dtype=np.uint64)
    codes = np.dot(encodings.astype(np.uint64), powers)

    return codes.astype(np.uint64)
