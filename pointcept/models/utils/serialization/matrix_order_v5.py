#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v5
# @Time          : 2025-07-31 00:18:48
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 移除了反转维度的空间填充曲线编码实现，保留了xyz的顺序
"""
import numpy as np
import time


def space_filling_encode_v5(
    coords: np.ndarray, M: np.ndarray, dim: int = 3, depth: int = 16
) -> np.ndarray:
    print(f"输入坐标形状: {coords.shape}, 维度: {dim}, 深度: {depth}")
    # 初始化时间统计字典（单位：毫秒）
    time_stats = {}

    # 步骤0: 输入处理
    start_time = time.perf_counter()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    N = coords.shape[0]
    shifts = np.arange(depth - 1, -1, -1)
    powers = 1 << np.arange(dim * depth - 1, -1, -1, dtype=np.uint64)
    time_stats["input_processing"] = (time.perf_counter() - start_time) * 1000

    # 步骤1: 向量化二进制转换
    start_time = time.perf_counter()
    coord_bits = ((coords[:, :, np.newaxis] >> shifts) & 1).astype(np.uint8)
    time_stats["binary_conversion"] = (time.perf_counter() - start_time) * 1000

    # 步骤2: 按z,y,x顺序拼接并reshape
    start_time = time.perf_counter()
    v = coord_bits.transpose(0, 2, 1).reshape(N, -1)  # 直接reshape，不再反转维度
    time_stats["bit_reshaping"] = (time.perf_counter() - start_time) * 1000

    # 步骤3: 矩阵运算
    start_time = time.perf_counter()
    encodings = np.bitwise_and(v @ M.T, 1)
    print(encodings[:1])  # 打印前5个编码以检查数据
    time_stats["matrix_operation"] = (time.perf_counter() - start_time) * 1000

    # 步骤4: 生成最终编码
    start_time = time.perf_counter()
    codes = np.dot(encodings.astype(np.uint64), powers)
    time_stats["final_encoding"] = (time.perf_counter() - start_time) * 1000

    # 总时间
    time_stats["total_time"] = sum(time_stats.values())

    # 打印时间统计（毫秒单位）
    print("\nTime Statistics (ms):")
    for step, duration in time_stats.items():
        print(f"{step}: {duration:.6f} ms")

    return codes.astype(np.uint64)
