#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v7
# @Time          : 2025-07-31 00:18:48
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 优化版空间填充曲线编码，通过逐步计算子矩阵加速，添加了查找表优化
"""
import numpy as np
import time


def build_lut_for_submatrix(M):
    """为子矩阵构建查找表"""
    # 计算输入位数
    input_bits = M.shape[1]
    # 生成所有可能的输入组合
    inputs = np.array(
        [
            [(i >> j) & 1 for j in range(input_bits - 1, -1, -1)]
            for i in range(1 << input_bits)
        ],
        dtype=np.uint8,
    )
    # 计算输出
    return np.mod(np.dot(inputs, M.T), 2)


def space_filling_encode_v7(
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
    # 初始化时间统计字典（单位：毫秒）
    time_stats = {}

    # 步骤0: 输入处理和预计算
    start_time = time.perf_counter()
    if hasattr(coords, "numpy"):
        coords = coords.numpy()
    N = coords.shape[0]
    print(
        f"输入坐标形状: {coords.shape}, 维度: {dim}, 深度: {depth}, 子块数量: {len(submatrices)}"
    )
    print(coords[:5])  # 打印前5个坐标以检查数据
    # 计算总深度和每个子块的位移
    total_depth = sum(level_list)

    shifts = np.arange(total_depth - 1, -1, -1)

    # 预计算2的幂次方
    powers = 1 << np.arange(dim * total_depth - 1, -1, -1, dtype=np.uint64)

    # 预构建所有子矩阵的查找表
    luts = [build_lut_for_submatrix(M) for M in submatrices]
    # print(submatrices[0])
    # for i in range(64):
    #     print(f"lut[1][{i}] = {luts[1][i]}")  # 打印luts[0]的前8个值
    time_stats["input_processing"] = (time.perf_counter() - start_time) * 1000

    # 步骤1: 向量化二进制转换
    start_time = time.perf_counter()
    coord_bits = ((coords[:, :, np.newaxis] >> shifts) & 1).astype(np.uint8)
    print(coord_bits[:1])  # 打印前5个坐标的二进制表示
    time_stats["binary_conversion"] = (time.perf_counter() - start_time) * 1000

    # 步骤2: reshape
    start_time = time.perf_counter()
    v = coord_bits.transpose(0, 2, 1).reshape(N, -1)  # 直接reshape，不再反转维度
    print(v[:1])  # 打印前5个重塑后的二进制表示
    time_stats["bit_reshaping"] = (time.perf_counter() - start_time) * 1000

    # 步骤3: 使用查找表进行分块矩阵运算
    start_time = time.perf_counter()
    encodings = np.zeros((N, dim * total_depth), dtype=np.uint8)

    bit_offset = 0
    for lut, r in zip(luts, level_list):
        block_size = dim * r
        current_block = v[:, bit_offset : bit_offset + block_size]

        # 计算输入索引
        input_indices = np.zeros(N, dtype=np.uint32)
        for i in range(block_size):
            input_indices |= current_block[:, i] << (block_size - 1 - i)

        # 使用查找表
        encodings[:, bit_offset : bit_offset + block_size] = lut[input_indices]
        bit_offset += block_size

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
