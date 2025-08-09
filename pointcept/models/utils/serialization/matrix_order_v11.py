#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本：使用稀疏矩阵加速计算（当M的稀疏性>70%时效率显著提升）
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix


def int_to_bits(n, bits=16):
    """保持原位运算实现"""
    return [(n >> i) & 1 for i in range(bits - 1, -1, -1)]


def space_filling_encode_batch_sparse(coords, M):
    """
    批量空间填充曲线编码（稀疏矩阵优化版）

    参数:
        coords: 坐标张量，形状为 (N, 3)，类型为np.ndarray或torch.Tensor
        M: 满秩矩阵（48×48），建议提前转换为CSR格式

    返回:
        codes: 编码数组，形状为 (N,)，64位无符号整数
    """
    # 输入统一处理
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = np.ascontiguousarray(coords, dtype=np.uint16)

    # 将M转换为CSR稀疏矩阵（若尚未转换）
    if not isinstance(M, csr_matrix):
        M_sparse = csr_matrix(M.astype(np.uint8))
    else:
        M_sparse = M

    sparsity = 1 - M_sparse.nnz / (M_sparse.shape[0] * M_sparse.shape[1])
    print(f"矩阵稀疏性：{sparsity:.1%}")  # >30%时推荐使用本函数

    N = coords.shape[0]
    v = np.zeros((N, 48), dtype=np.uint8)

    # 向量化位填充（与原函数相同）
    for i in range(16):
        v[:, 15 - i] = (coords[:, 2] >> i) & 1
        v[:, 31 - i] = (coords[:, 1] >> i) & 1
        v[:, 47 - i] = (coords[:, 0] >> i) & 1

    # 稀疏矩阵乘法（模2）
    encodings = (v @ M_sparse.T) % 2  # 自动利用稀疏性

    # 位打包（保持原高效方法）
    codes = np.zeros(N, dtype=np.uint64)
    for i in range(6):
        start = i * 8
        end = start + 8
        codes |= np.left_shift(
            np.packbits(encodings[:, start:end], axis=1, bitorder="big")
            .astype(np.uint64)
            .flatten(),
            (5 - i) * 8,
        )

    return codes
