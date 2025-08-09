#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v8
# @Time          : 2025-07-31 15:21:36
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 优化版空间填充曲线编码，通过逐步计算子矩阵加速，添加了查找表优化(only for eqal submatrices)
"""
import numpy as np
import time


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


def space_filling_encode_v8(
    coords: np.ndarray, dim: int, depth: int, level_list: list, submatrices: list
) -> np.ndarray:
    """
    优化版空间填充曲线编码，通过逐步计算子矩阵加速，添加了查找表优化

    参数:
        coords: 输入坐标，形状为(N, dim)
        dim: 坐标维度 (d)
        depth: 总深度
        level_list: 各子块对应的层数列表 [r1, r2, ...]
        submatrices: 子矩阵列表，每个形状为(dim*r_i) × (dim*r_i)

    返回:
        编码后的空间填充曲线值
    """
    r_len = len(level_list)
    time_stats = {}

    # 步骤0: 输入处理和预计算
    start_time = time.perf_counter()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()

    coords = coords.astype(np.uint64)
    N = coords.shape[0]

    # 预构建所有子矩阵的查找表
    luts = [build_lut_for_submatrix(M) for M in submatrices]
    # 循环遍历luts[0]的值，从0~7
    # for i in range(64):
    #     print(f"lut[1][{i}] = {luts[1][i]}")  # 打印luts[0]的前8个值

    time_stats["input_processing"] = (time.perf_counter() - start_time) * 1000

    # 步骤1: 位提取与LUT索引计算
    start_time = time.perf_counter()
    input_indices = np.zeros((N, r_len), dtype=np.uint64)
    done_bits = 0
    for i in range(r_len):
        r = level_list[i]

        input_indices[:, i] = 0
        for j in range(r):
            bit_pos = depth - 1 - (done_bits + j)  # 从高位向低位取
            shift = (r - 1 - j) * 3
            input_indices[:, i] |= (
                (((coords[:, 0] >> bit_pos) & 1) << (2 + shift))
                | (((coords[:, 1] >> bit_pos) & 1) << (1 + shift))
                | (((coords[:, 2] >> bit_pos) & 1) << shift)
            )
        done_bits += r
    print(input_indices[:1])  # 打印前5个输入索引
    time_stats["bit_extraction"] = (time.perf_counter() - start_time) * 1000

    # 步骤2: 使用查找表进行分块矩阵运算
    start_time = time.perf_counter()
    results = np.zeros((N, r_len), dtype=np.uint16)

    for i in range(r_len):
        results[:, i] = luts[i][input_indices[:, i]]
    print(results[:1])  # 打印前5个结果
    time_stats["matrix_operation"] = (time.perf_counter() - start_time) * 1000

    # 步骤3: 生成最终编码
    start_time = time.perf_counter()
    bit_contributions = [level * dim for level in level_list]
    shifts = np.cumsum(bit_contributions[::-1])[::-1]
    shifts = np.roll(shifts, -1)
    shifts[-1] = 0
    codes = np.zeros(N, dtype=np.uint64)
    for i in range(r_len):
        codes |= results[:, i].astype(np.uint64) << shifts[i].astype(np.uint64)
    time_stats["final_encoding"] = (time.perf_counter() - start_time) * 1000

    # 总时间
    time_stats["total_time"] = sum(time_stats.values())

    print("\nTime Statistics (ms):")
    for step, duration in time_stats.items():
        print(f"{step}: {duration:.6f} ms")

    return codes.astype(np.uint64)
