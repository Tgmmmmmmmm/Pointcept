import torch
import numpy as np


def build_lut_for_submatrix(M):
    """构建子矩阵的查找表（LUT），支持 GPU/CPU。"""
    n = M.size(0)  # 子矩阵大小
    assert 1 <= n <= 16, "n必须在1~16范围内"
    device = M.device  # 继承输入张量的设备

    total_inputs = 1 << n  # 2^n
    i = torch.arange(total_inputs, dtype=torch.int32, device=device).view(-1, 1)
    k = torch.arange(n, dtype=torch.int8, device=device)

    # 生成所有输入的二进制位 (total_inputs × n)，并转换为和 M 相同的数据类型
    bits = ((i >> (n - 1 - k)) & 1).to(M.dtype)  # 强制转换为 M.dtype

    # 计算矩阵乘法 bits @ M^T，并取模2
    result = (bits @ M.T) & 1  # shape: [total_inputs, n]

    # 将结果打包成 uint16（最高位在左）
    shifts = torch.arange(n - 1, -1, -1, dtype=torch.int16, device=device)
    lut = (result.to(torch.int16) << shifts).sum(dim=1, dtype=torch.int16)
    return lut


def space_filling_encode_v12(coords, dim, depth, level_list, submatrices):
    """空间填充编码，支持 PyTorch 张量输入（CPU/GPU）。"""
    device = coords.device  # 继承输入张量的设备

    # 确保 submatrices 是 torch.Tensor 列表
    if isinstance(submatrices, list):
        submatrices = [
            torch.from_numpy(M) if isinstance(M, np.ndarray) else M for M in submatrices
        ]
    elif isinstance(submatrices, np.ndarray):
        submatrices = torch.from_numpy(submatrices)

    # 将所有子矩阵移动到正确设备
    luts = [build_lut_for_submatrix(M.to(device)) for M in submatrices]

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
        codes |= luts[i][input_indices].to(torch.int64) << (code_shifts[i] * dim)

    return codes
