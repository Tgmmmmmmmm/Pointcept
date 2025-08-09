#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : diy_order
# @Time          : 2025-08-05 00:22:21
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 初版,最小且相同模块
# @version       : 0.1
"""
import torch
from typing import Optional, Union
from collections import defaultdict


# 查找表，根据规则矩阵和设备


class MatrixLUT:
    def __init__(self, n=3):
        """初始化查找表，默认支持z-order的3x3矩阵"""
        self._encode = {}  # 初始化存储LUT的字典
        device = torch.device("cpu")
        # 确保设备键存在
        if device not in self._encode:
            self._encode[device] = {}
        z_order = torch.diag(torch.ones(n, dtype=torch.int32)).fliplr()
        self._encode[device][self.matrix_to_key(z_order)] = self.compute_lut(z_order)

    def encode_lut(self, matrix, device=torch.device("cpu")):
        """获取指定设备上的矩阵LUT（自动处理设备迁移）"""
        key = self.matrix_to_key(matrix)

        # 初始化设备条目（如果不存在）
        if device not in self._encode:
            self._encode[device] = {}

        # 缓存未命中时计算并存储
        # TODO ： 目前没有判断矩阵满秩，需要传入时确定
        if key not in self._encode[device]:
            # assert torch.linalg.matrix_rank(matrix) == matrix.size(
            #     0
            # ), "Matrix must be full-rank"
            lut = self.compute_lut(matrix).to(device)
            self._encode[device][key] = lut

        return self._encode[device][key]

    # 将矩阵转换为键,适用在12x12以下
    @staticmethod
    def matrix_to_key(matrix):
        """将矩阵转换为键，适用于12x12以下的矩阵"""
        if isinstance(matrix, torch.Tensor):
            # 如果是张量，转换为嵌套元组
            return tuple(tuple(row.tolist()) for row in matrix)
        elif isinstance(matrix, (list, tuple)):
            # 如果是列表或元组，直接转换为嵌套元组
            return tuple(
                tuple(row) if hasattr(row, "__iter__") else (row,) for row in matrix
            )
        else:
            # 其他情况（如单个数值），包装为单元素元组
            return ((matrix,),)

    def compute_lut(self, M):
        """计算查找表，输入M是一个维度小于等于16的满秩方阵"""

        n = M.size(0)
        assert 1 <= n <= 16, "n must be in 1-16"
        assert M.size(1) == n, "M must be a square matrix"

        # 如果已经存在这个矩阵的查找表，直接返回
        if self.matrix_to_key(M) in self._encode[M.device]:
            return self._encode[self.matrix_to_key(M)]
        # assert torch.linalg.matrix_rank(M) == n, "M must be a full-rank matrix"

        # 根据 n 选择 dtype
        if n <= 6:
            dtype = torch.int8  # 节省内存
        else:
            dtype = torch.int16  # 安全覆盖 n=7..16
        M = M.to(dtype)  # 确保输入是整数类型
        # 计算 LUT
        bits = (
            torch.arange(1 << n, dtype=dtype).view(-1, 1)
            >> torch.arange(n - 1, -1, -1, dtype=torch.int8)
        ) & 1
        lut = ((bits @ M.T) & 1).to(dtype) << torch.arange(
            n - 1, -1, -1, dtype=torch.int8
        )
        return lut.sum(dim=1, dtype=dtype)


_matrix_key = MatrixLUT()


# version 1 , only fixed size, fixed rule_matrix
def matrix2key(coords, dim, depth, level_list, submatrices):
    """空间填充编码，支持 PyTorch 张量输入（CPU/GPU）。"""
    device = coords.device  # 继承输入张量的设备
    N = coords.size(0)
    r_len = len(level_list)

    # 预计算位移量（GPU兼容）
    bit_shifts = torch.zeros(r_len, dtype=torch.uint8, device=device)
    code_shifts = torch.zeros(r_len, dtype=torch.uint8, device=device)
    coord_shift = 0
    for i in range(r_len - 1, -1, -1):
        code_shifts[i] = coord_shift
        coord_shift += level_list[i]
        bit_shifts[i] = coord_shift

    codes = torch.zeros(N, dtype=torch.int64, device=device)

    for i, r in enumerate(level_list):
        input_indices = torch.zeros(N, dtype=torch.int32, device=device)
        for j in range(r):
            bit_pos = bit_shifts[i] - 1 - j
            # 提取所有维度的位
            bits = ((coords >> bit_pos) & 1).to(torch.int32)
            j_shift = (r - j - 1) * dim
            # 组合成 dim 位一组
            for d in range(dim):
                input_indices |= bits[:, d] << (dim - 1 - d + j_shift)

        # 使用LUT查找并组合结果
        codes |= _matrix_key.encode_lut(submatrices[i], device=device)[
            input_indices
        ].to(torch.int64) << (code_shifts[i] * dim)

    return codes
