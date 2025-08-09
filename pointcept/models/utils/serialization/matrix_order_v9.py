#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v9
# @Time          : 2025-08-02 16:47:08
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 最优化版本，空间填充曲线按子矩阵利用查找表
"""
import numpy as np


def build_lut_for_submatrix(M):
    n = M.shape[0]  # 获取子矩阵的大小
    assert 1 <= n <= 16, "n必须在2~16范围内"
    total_inputs = 1 << n  # 2^n

    # 生成所有输入 i 的二进制位矩阵 (total_inputs × n)
    i = np.arange(total_inputs, dtype=np.uint16).reshape(-1, 1)
    k = np.arange(n, dtype=np.uint8)
    bits = (i >> (n - 1 - k)) & 1  # shape: (total_inputs, n)

    # 计算矩阵乘法 bits @ M，并取模2
    result = (bits @ M.T) & 1  # shape: (total_inputs, n)

    # 将结果打包成 uint16（最高位在左）
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint16)
    lut = np.left_shift(result.astype(np.uint16), shifts).sum(axis=1, dtype=np.uint16)
    return lut


def space_filling_encode_v9(coords, dim, depth, level_list, submatrices):
    luts = [build_lut_for_submatrix(M) for M in submatrices]
    N = coords.shape[0]
    r_len = len(level_list)
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    # 预计算所有位移量
    bit_shifts = np.zeros(r_len, dtype=np.uint8)
    code_shifts = np.zeros(r_len, dtype=np.uint8)
    coord_shift = 0
    for i in range(r_len - 1, -1, -1):
        code_shifts[i] = coord_shift
        coord_shift += level_list[i]
        bit_shifts[i] = coord_shift

    # 使用uint32足够，因为最大深度通常不会超过32
    codes = np.zeros(N, dtype=np.uint64)

    # 一次性处理所有坐标点
    for i, r in enumerate(level_list):
        input_indices = np.zeros(N, dtype=np.uint32)
        for j in range(r):
            bit_pos = bit_shifts[i] - 1 - j
            # 同时提取三个坐标的位
            bits = ((coords >> bit_pos) & 1).astype(np.uint32)
            j_shift = (r - j - 1) * 3
            # 组合成3位一组
            input_indices |= (
                (bits[:, 0] << (2 + j_shift))
                | (bits[:, 1] << (1 + j_shift))
                | (bits[:, 2] << (j_shift))
            )
        # 使用LUT查找并组合结果
        codes |= luts[i][input_indices].astype(np.uint64) << (code_shifts[i] * dim)

    return codes
