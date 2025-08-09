#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : matrix_order_v0
# @Time          : 2025-07-30 09:02:06
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : 朴素版z-order空间填充曲线编码实现
"""
import numpy as np


def morton_naive(x, y, z, depth=16):
    """可配置深度的朴素Morton码计算（单个点）"""
    result = 0
    for i in range(depth):  # 处理指定深度的位数
        result |= ((z >> i) & 1) << (3 * i)
        result |= ((y >> i) & 1) << (3 * i + 1)
        result |= ((x >> i) & 1) << (3 * i + 2)
    return result


def space_filling_encode_v0(coords, depth=16):
    """批量计算可配置深度的Morton码（朴素方法）"""
    return np.array(
        [morton_naive(x, y, z, depth) for x, y, z in coords], dtype=np.uint64
    )
