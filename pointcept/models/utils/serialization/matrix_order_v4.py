#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v4
# @Time          : 2025-07-30 09:28:54
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 统计v3的各个步骤时间
"""
import numpy as np
import time

# .预计算位移量耗时: 0.004ms
# 向量化二进制转换耗时: 8.224ms
# 按z,y,x顺序拼接耗时: 1.039ms
# 矩阵运算耗时: 67.569ms
# 生成最终编码耗时: 6.003ms


def space_filling_encode_v4(
    coords: np.ndarray, M: np.ndarray, dim: int = 3, depth: int = 16
) -> np.ndarray:
    N = coords.shape[0]
    total_bits = dim * depth
    # 如果输入是 PyTorch Tensor，转换为 NumPy 数组
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    # 1. 预计算位移量
    start = time.time()
    shifts = np.arange(depth - 1, -1, -1)
    time_shifts = (time.time() - start) * 1000  # 转换为毫秒

    # 2. 向量化二进制转换
    start = time.time()
    coord_bits = ((coords[:, :, np.newaxis] >> shifts) & 1).astype(np.uint8)
    time_binary = (time.time() - start) * 1000  # 转换为毫秒

    # 3. 按z,y,x顺序拼接
    start = time.time()
    v = coord_bits.transpose(0, 2, 1).reshape(N, -1)
    time_reshape = (time.time() - start) * 1000  # 转换为毫秒

    # 4. 矩阵运算
    start = time.time()
    encodings = (v @ M.T) % 2
    time_matrix = (time.time() - start) * 1000  # 转换为毫秒

    # 5. 生成最终编码
    start = time.time()
    powers = 1 << np.arange(total_bits - 1, -1, -1, dtype=np.uint64)
    codes = np.dot(encodings.astype(np.uint64), powers)
    time_final = (time.time() - start) * 1000  # 转换为毫秒

    # 打印各步骤耗时（毫秒单位）
    print(f"\n预计算位移量耗时: {time_shifts:.3f}ms")
    print(f"向量化二进制转换耗时: {time_binary:.3f}ms")
    print(f"按z,y,x顺序拼接耗时: {time_reshape:.3f}ms")
    print(f"矩阵运算耗时: {time_matrix:.3f}ms")
    print(f"生成最终编码耗时: {time_final:.3f}ms")

    return codes.astype(np.uint64)
