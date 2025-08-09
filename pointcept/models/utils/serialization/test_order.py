import pytest
import torch
import time
import numpy as np

from pointcept.models.utils.serialization.matrix_order_v0 import space_filling_encode_v0
from pointcept.models.utils.serialization.matrix_order_v1 import space_filling_encode_v1

from pointcept.models.utils.serialization.matrix_order_v10 import (
    space_filling_encode_v10,
)
from pointcept.models.utils.serialization.matrix_order_v12 import (
    space_filling_encode_v12,
)
from pointcept.models.utils.serialization.matrix_order_v2 import space_filling_encode_v2
from pointcept.models.utils.serialization.matrix_order_v3 import space_filling_encode_v3
from pointcept.models.utils.serialization.matrix_order_v4 import space_filling_encode_v4
from pointcept.models.utils.serialization.matrix_order_v5 import space_filling_encode_v5
from pointcept.models.utils.serialization.matrix_order_v6 import space_filling_encode_v6
from pointcept.models.utils.serialization.matrix_order_v7 import (
    space_filling_encode_v7,
)
from pointcept.models.utils.serialization.matrix_order_v8 import (
    space_filling_encode_v8,
)
from pointcept.models.utils.serialization.matrix_order_v9 import (
    space_filling_encode_v9,
)
from .default import encode, decode, z_order_encode, hilbert_encode, diy_order_encode

# from .matrix_order_v2 import space_filling_encode_v2
# from .matrix_order_v3 import space_filling_encode_v3
# from .matrix_order_v4 import space_filling_encode_v4
# from .matrix_order_v5 import space_filling_encode_v5
# from .matrix_order_v6 import space_filling_encode_v6
# from .matrix_order_v7 import space_filling_encode_v7
# from .matrix_order_v8 import space_filling_encode_v8
# from .matrix_order_v9 import space_filling_encode_v9
# from .matrix_order_v10 import space_filling_encode_v10
# from .matrix_order_v11 import space_filling_encode_v11


def generate_full_rank_matrix(dim, level, seed=None, M=None):
    """
    生成或验证一个 (dim*level) × (dim*level) 的二元满秩方阵(F_2下)

    参数:
        dim: 坐标维度 (d)
        level: 划分层数 (r)
        seed: 随机种子，None表示不固定
        M: 直接提供的矩阵，如果非None则只验证不满秩报错

    返回:
        生成的矩阵或验证通过的矩阵
    """
    size = dim * level

    if M is not None:
        # 验证模式
        if M.shape != (size, size):
            raise ValueError(f"提供的矩阵形状{M.shape}不符合要求的({size}, {size})")
        if np.linalg.matrix_rank(M.astype(np.float32)) != size:
            raise ValueError("提供的矩阵不是满秩的")
        return M

    if seed is not None:
        np.random.seed(seed)

    # 生成矩阵直到找到满秩的
    while True:
        M = np.random.randint(0, 2, (size, size), dtype=np.uint8)
        if np.linalg.matrix_rank(M.astype(np.float32)) == size:
            return M


def create_block_diagonal(
    dim, level_list=None, total_level=None, submatrices=None, seed=None
):
    """
    通过块对角方法创建 (dim*total_level) × (dim*total_level) 的矩阵

    参数:
        dim: 坐标维度 (d)
        level_list: 各子块的层数列表 [r1, r2, ...]
        total_level: 总层数，如果不为None则验证sum(level_list)==total_level
        submatrices: 提供的子矩阵列表，每个子矩阵应为 (dim*r_i) × (dim*r_i)
        seed: 随机种子，None表示不固定

    返回:
        生成的块对角矩阵和各子块矩阵
    """

    if total_level is None:
        raise ValueError("必须提供total_level参数")

    # 处理level_list为None或空的情况
    if level_list is None or len(level_list) == 0:
        level_list = [1] * total_level

    # 计算level_list的总和
    sum_level = sum(level_list)

    # 验证总和是否匹配total_level
    if sum_level != total_level:
        raise ValueError(
            f"level_list的总和{sum_level}不等于指定的total_level{total_level}"
        )

    size = dim * total_level

    if seed is not None:
        np.random.seed(seed)

    if submatrices is None:
        submatrices = []
        for i, r in enumerate(level_list):
            sub_size = dim * r
            # 为每个子块生成满秩矩阵
            submatrices.append(
                generate_full_rank_matrix(
                    dim=dim, level=r, seed=seed + i if seed is not None else None
                )
            )
    else:
        # 验证提供的子矩阵
        if len(submatrices) != len(level_list):
            raise ValueError("提供的子矩阵数量与level_list长度不匹配")
        for i, (M, r) in enumerate(zip(submatrices, level_list)):
            expected_size = dim * r
            if M.shape != (expected_size, expected_size):
                raise ValueError(
                    f"子矩阵{i}的形状{M.shape}与要求的({expected_size}, {expected_size})不匹配"
                )
            generate_full_rank_matrix(dim=dim, level=r, M=M)  # 验证满秩

    # 创建块对角矩阵
    block_diag = np.zeros((size, size), dtype=np.uint8)

    pos = 0
    for M, r in zip(submatrices, level_list):
        sub_size = dim * r
        block_diag[pos : pos + sub_size, pos : pos + sub_size] = M
        pos += sub_size

    return block_diag, submatrices, level_list


def print_stats(title, data, time_taken=None, sample_count=6, unit="ms"):
    """打印统计信息的通用函数（增强可读性版本）

    Args:
        title (str): 测试标题
        data: 要统计的数据（通常是numpy数组或张量）
        time_taken (float, optional): 耗时（秒）
        sample_count (int): 要显示的样本数量
        unit (str): 时间单位 ('ms' 或 's')
    """
    # 使用分隔线增强可读性
    separator = "=" * 100
    sub_separator = "-" * 100

    print(f"\n{separator}")
    print(f"=== {title.upper()} ===".center(100))
    print(separator)

    if time_taken is not None:
        if unit.lower() == "ms":
            time_str = f"{time_taken * 1000:.2f} 毫秒"
        else:
            time_str = f"{time_taken:.4f} 秒"
        print(f"🕒 耗时: {time_str}")
        print(sub_separator)

    print(f"📐 数据形状: {data.shape}")
    print(sub_separator)

    # 样本数据打印
    print(f"🔍 样本值（前 {sample_count} 个）:")
    print(sub_separator)
    print(data[:sample_count])
    print(f"{separator}\n")


class TestSpaceFillingCurves:
    """空间填充曲线测试类，统一管理测试数据和测试方法"""

    @classmethod
    def setup_class(cls):
        """初始化测试数据，所有测试共用同一份数据"""
        start_time = time.perf_counter()
        cls.dim = 3  # 坐标维度
        cls.depth = 16  # 树的最大深度
        cls.level_list = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]  # 每层的划分数量，默认为1
        assert sum(cls.level_list) == cls.depth, "level_list总和必须等于depth"
        # 生成满秩矩阵和块对角矩阵
        cls.M = generate_full_rank_matrix(dim=cls.dim, level=cls.depth, seed=42)
        cls.M_diag, cls.submatrices, cls.level_list = create_block_diagonal(
            dim=cls.dim, level_list=cls.level_list, total_level=cls.depth, seed=42
        )
        cls.print_matrices()

        cls.batch_size = 32  # 批次数量
        cls.points_per_batch = 3200  # 每批的点数
        cls.num_points = cls.batch_size * cls.points_per_batch  # 总点数 = 32 * 3200
        # 根据深度计算最大坐标值，确保在2^depth范围内
        cls.max_val = 2**cls.depth - 1  # 65535 for depth=16

        # 生成固定随机种子的大规模测试数据
        np.random.seed(42)
        coords = np.random.randint(0, cls.max_val, size=(cls.num_points, cls.dim))
        cls.coords = torch.tensor(coords, dtype=torch.long)

        # 生成批次信息（每个批次102400个点）
        cls.batch = torch.repeat_interleave(
            torch.arange(cls.batch_size), cls.points_per_batch
        )

        cls.diy_order = torch.diag(torch.ones(3, dtype=torch.int32))

        print_stats("初始化测试数据", cls.coords, time.perf_counter() - start_time)
        print(
            f"生成 {cls.num_points} 个测试点（{cls.batch_size} 批，每批 {cls.points_per_batch} 个点），坐标范围 0-{cls.max_val}"
        )

    @classmethod
    def print_matrices(cls):
        print("\n" + "=" * 100)
        print(f"{'矩阵生成结果':^100}")
        print(f"{f'维度(dim): {cls.dim}, 层数(depth): {cls.depth}':^100}")
        print("=" * 100)

        print("\n1. 普通满秩矩阵 cls.M:")
        print(f"形状: {cls.M.shape}")
        print("矩阵内容:")
        print(cls.M)

        print("\n2. 块对角矩阵 cls.M_diag:")
        print(f"形状: {cls.M_diag.shape}")
        print("矩阵内容:")
        print(cls.M_diag)

        print("\n3. 子矩阵列表 cls.submatrices:")
        for i, submatrix in enumerate(cls.submatrices):
            print(f"\n子矩阵 {i+1}:")
            print(f"形状: {submatrix.shape}")
            print(submatrix)

    def test_hilbert_encode(self):
        """测试Hilbert编码"""
        start_time = time.perf_counter()
        codes = hilbert_encode(self.coords, depth=self.depth)
        print_stats("Hilbert编码测试", codes, time.perf_counter() - start_time)

    def test_naive_z_order_encode(self):
        """测试朴素 Z-order编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v0(self.coords.numpy(), self.depth)
        print_stats("朴素Z-order编码测试", codes, time.perf_counter() - start_time)

    def test_z_order_encode(self):
        """测试 LUT Z-order编码"""
        start_time = time.perf_counter()
        codes = z_order_encode(self.coords, depth=self.depth)
        print_stats("LUT-Z-order编码测试", codes, time.perf_counter() - start_time)

    def test_diy_order_encode(self):
        """测试 diy Z-order编码"""
        start_time = time.perf_counter()
        codes = diy_order_encode(self.coords, self.diy_order, depth=self.depth)
        print_stats("LUT-Z-order编码测试", codes, time.perf_counter() - start_time)

    @pytest.mark.skip(reason="跳过这个测试，原因：性能非常差, 运行的太久了！")
    def test_matrix_order_v1_encode(self):
        """测试 Matrix_order_v1 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v1(self.coords, self.M, self.dim, self.depth)
        print_stats("Matrix_order_v1 编码测试", codes, time.perf_counter() - start_time)

    def test_matrix_order_v2_encode(self):
        """测试 Matrix_order_v2 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v2(self.coords, self.M, self.dim, self.depth)
        print_stats(
            "Matrix_order_v2 (向量化) 编码测试", codes, time.perf_counter() - start_time
        )

    def test_matrix_order_v3_encode(self):
        """测试 Matrix_order_v3 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v3(self.coords, self.M, self.dim, self.depth)
        print_stats(
            "Matrix_order_v3 (位运算) 编码测试", codes, time.perf_counter() - start_time
        )

    def test_matrix_order_v4_encode(self):
        """测试 Matrix_order_v4 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v4(self.coords, self.M, self.dim, self.depth)
        print_stats(
            "Matrix_order_v4 (v3时间统计) 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    def test_matrix_order_v5_encode(self):
        """测试 Matrix_order_v5 编码"""
        start_time = time.perf_counter()
        print(self.dim, self.depth, self.M.shape)
        codes = space_filling_encode_v5(self.coords, self.M, self.dim, self.depth)
        print_stats(
            "Matrix_order_v5 (xyz顺序) 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    def test_matrix_order_v6_encode(self):
        """测试 Matrix_order_v6 编码"""
        start_time = time.perf_counter()
        codes_all_mat = space_filling_encode_v5(
            self.coords, self.M_diag, self.dim, self.depth
        )
        print_stats(
            "Matrix_order_v5 完整矩阵 编码测试",
            codes_all_mat,
            time.perf_counter() - start_time,
        )

        start_time = time.perf_counter()
        codes_sub_mat = space_filling_encode_v6(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        print_stats(
            "Matrix_order_v6 子矩阵 编码测试",
            codes_sub_mat,
            time.perf_counter() - start_time,
        )
        assert (
            codes_all_mat.shape == codes_sub_mat.shape
        ), f"编码形状不匹配: {codes_all_mat.shape} != {codes_sub_mat.shape}"
        assert np.array_equal(
            codes_all_mat, codes_sub_mat
        ), "编码结果不一致: 全矩阵与子矩阵编码结果不同"

    def test_matrix_order_v7_encode(self):
        """测试 Matrix_order_v7 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v7(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )

        print_stats(
            "Matrix_order_v7 (子矩阵查找表优化) 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    def test_matrix_order_v8_encode(self):
        """测试 Matrix_order_v8 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v8(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        print_stats(
            "Matrix_order_v8 (向量化查找表) 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    def test_matrix_order_v9_encode(self):
        """测试 Matrix_order_v9 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v9(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        print_stats(
            "Matrix_order_v9 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    # def test_matrix_order_v9_encode_z_order(self):
    #     """测试 Matrix_order_v9 Z-order编码"""
    #     start_time = time.perf_counter()
    #     codes = codes = z_order_encode(self.coords, depth=self.depth)
    #     print_stats(
    #         "Matrix_order_v9 Z-order LUT_orgin编码测试",
    #         codes,
    #         time.perf_counter() - start_time,
    #     )
    #     # 定义矩阵大小
    #     size = 3

    #     # 生成一个 3×3 反对角线矩阵（反主对角线全为1，其余为0）
    #     M = np.zeros((size, size), dtype=np.uint8)  # 初始化为全0矩阵
    #     # np.fill_diagonal(np.fliplr(M), 1)  # 填充反对角线为1
    #     np.fill_diagonal(M, 1)  # 填充主对角线为1
    #     print("单个反对角线矩阵：")
    #     print(M)

    #     # 生成16个相同的反对角线矩阵，并存储在一个列表中
    #     matrix_list = [M.copy() for _ in range(self.depth)]
    #     start_time = time.perf_counter()

    #     codes = space_filling_encode_v9(
    #         self.coords, self.dim, self.depth, self.level_list, matrix_list
    #     )
    #     print_stats(
    #         "Matrix_order_v9 Z-order编码测试",
    #         codes,
    #         time.perf_counter() - start_time,
    #     )
    #     all_matrix, _, _ = create_block_diagonal(
    #         dim=self.dim,
    #         level_list=self.level_list,
    #         total_level=self.depth,
    #         submatrices=matrix_list,
    #     )

    #     start_time = time.perf_counter()
    #     codes = space_filling_encode_v5(self.coords, all_matrix, self.dim, self.depth)
    #     print_stats(
    #         "Matrix_order_v5 Z-order编码测试",
    #         codes,
    #         time.perf_counter() - start_time,
    #     )

    def test_matrix_order_v10(self):
        start_time = time.perf_counter()
        codes = space_filling_encode_v10(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        print_stats(
            "Matrix_order_v10 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    # @pytest.mark.skip(reason="关闭对比测试")
    def test_matrix_order_compare(self):
        """比较不同版本的矩阵编码性能"""
        print("\n=== 矩阵编码性能对比测试 ===")
        # start_time = time.time()

        # # 测试 v0-v6 的编码
        # v0_codes = space_filling_encode_v0(self.coords.numpy(), self.depth)
        # print_stats("v0 编码", v0_codes, time.time() - start_time)

        # start_time = time.time()
        # v1_codes = space_filling_encode_v1(self.coords, self.M, self.dim, self.depth)
        # print_stats("v1 编码", v1_codes, time.time() - start_time)
        start_time = time.time()
        v2_codes = space_filling_encode_v2(self.coords, self.M, self.dim, self.depth)
        timev2 = time.time() - start_time
        print_stats("v2 编码", v2_codes, timev2)

        start_time = time.time()
        v3_codes = space_filling_encode_v3(self.coords, self.M, self.dim, self.depth)
        timev3 = time.time() - start_time
        print_stats("v3 编码", v3_codes, timev3)

        start_time = time.time()
        v4_codes = space_filling_encode_v4(self.coords, self.M, self.dim, self.depth)
        timev4 = time.time() - start_time
        print_stats("v4 编码", v4_codes, timev4)

        start_time = time.time()
        v5_codes = space_filling_encode_v5(self.coords, self.M, self.dim, self.depth)
        timev5 = time.time() - start_time
        print_stats("v5 编码", v5_codes, timev5)

        start_time = time.time()
        v6_codes_all = space_filling_encode_v5(
            self.coords, self.M_diag, self.dim, self.depth
        )
        timev6_all = time.time() - start_time
        print_stats("v6 全矩阵编码", v6_codes_all, timev6_all)

        start_time = time.time()
        v6_codes = space_filling_encode_v6(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        timev6 = time.time() - start_time
        print_stats("v6 编码", v6_codes, timev6)

        start_time = time.time()
        v7_codes = space_filling_encode_v7(
            self.coords,
            self.dim,
            self.depth,
            self.level_list,
            self.submatrices,
        )
        timev7 = time.time() - start_time
        print_stats("v7 编码", v7_codes, timev7)

        start_time = time.time()
        v8_codes = space_filling_encode_v8(
            self.coords, self.dim, self.depth, self.level_list, self.submatrices
        )
        timev8 = time.time() - start_time
        print_stats("v8 编码", v8_codes, timev8)

        start_time = time.time()
        v9_codes = space_filling_encode_v9(
            self.coords, self.dim, self.depth, self.level_list, self.submatrices
        )
        timev9 = time.time() - start_time
        print_stats("v9 编码", v9_codes, timev9)

        start_time = time.time()
        v10_codes = space_filling_encode_v10(
            self.coords, self.dim, self.depth, self.level_list, self.submatrices
        )
        timev10 = time.time() - start_time
        print_stats("v10 编码", v10_codes, timev10)

        # 打印时间对比

        assert (
            v2_codes.shape == v3_codes.shape == v4_codes.shape == v5_codes.shape
        ), "v2 v3 v4 整个矩形编码结果形状不一致"
        assert (
            v6_codes.shape == v7_codes.shape == v8_codes.shape
        ), "v6 v7 v8 子矩阵编码结果形状不一致"
        assert (
            np.array_equal(v2_codes, v3_codes)
            and np.array_equal(v3_codes, v4_codes)
            and np.array_equal(v4_codes, v5_codes)
        ), "编码结果不一致"

        assert np.array_equal(
            v6_codes_all, v6_codes
        ), "v6 全矩阵编码结果与子矩阵编码结果不一致"

        assert np.array_equal(v6_codes, v7_codes), "v6 v7 编码结果不一致"
        assert np.array_equal(v7_codes, v8_codes), "v7 v8 编码结果不一致"
        assert np.array_equal(v8_codes, v9_codes), "v8 v9 编码结果不一致"
        assert np.array_equal(v9_codes, v10_codes), "v9 v10 编码结果不一致"
        print(
            f"编码时间对比 (单位: 秒):\n"
            f"v2: {timev2*1000:.2f} ms\n"
            f"v3: {timev3*1000:.2f} ms\n"
            f"v4: {timev4*1000:.2f} ms\n"
            f"v5: {timev5*1000:.2f} ms\n"
            f"v6 全矩阵: {timev6_all*1000:.2f} ms\n"
            f"v6 子矩阵: {timev6*1000:.2f} ms\n"
            f"v7: {timev7*1000:.2f} ms\n"
            f"v8: {timev8*1000:.2f} ms\n"
            f"v9: {timev9*1000:.2f} ms\n"
            f"v10: {timev10*1000:.2f} ms\n"
        )
        print(
            timev2 * 1000,
            timev3 * 1000,
            timev4 * 1000,
            timev5 * 1000,
            timev6 * 1000,
            timev7 * 1000,
            timev8 * 1000,
            timev9 * 1000,
            timev10 * 1000,
        )

    def test_matrix_order_v12_encode(self):
        """测试 Matrix_order_v12 编码"""
        start_time = time.perf_counter()
        codes = space_filling_encode_v12(
            self.coords, self.dim, self.depth, self.level_list, self.submatrices
        )
        print_stats(
            "Matrix_order_v12 编码测试",
            codes,
            time.perf_counter() - start_time,
        )

    # def test_sparese_order_encode(self):
    #     """测试稀疏矩阵填充曲线编码"""
    #     print("\n=== 稀疏矩阵填充曲线编码测试 ===")
    #     start_time = time.time()

    #     M, submatrices = create_48x48_block_diagonal()

    #     # 批量编码
    #     # encodings = space_filling_encode_batch(self.coords, M)
    #     codes = space_filling_encode_batch_sparse(self.coords, M)
    #     print(f"耗时: {time.time() - start_time:.4f}秒")
    #     sample_encodings = codes[:6]
    #     print(f"编码完成，形状: {codes.shape}")
    #     print(f"样本编码值:\n{sample_encodings}")

    # def test_diy_order_encode(self):
    #     """测试自定义空间填充曲线编码"""
    #     print("\n=== 自定义空间填充曲线编码测试 ===")
    #     start_time = time.time()

    #     # 生成满秩矩阵
    #     # M = generate_full_rank_matrix(48)
    #     M, submatrices = create_48x48_block_diagonal()
    #     print(f"生成满秩矩阵，形状: {M.shape}")
    #     print(M)
    #     print(
    #         f"子矩阵数量: {len(submatrices)}，每个子矩阵形状: {[m.shape for m in submatrices]}"
    #     )
    #     # 批量编码
    #     # encodings = space_filling_encode_batch(self.coords, M)
    #     codes = space_filling_encode_batch_optimized(self.coords, submatrices)
    #     print(f"耗时: {time.time() - start_time:.4f}秒")
    #     sample_encodings = codes[:6]
    #     print(f"编码完成，形状: {codes.shape}")
    #     print(f"样本编码值:\n{sample_encodings}")

    # def test_batch_handling(self):
    #     """测试批量处理功能"""
    #     print(f"\n=== 批量处理测试 ({self.batch_size}批次) ===")

    #     # Z-order批量测试
    #     z_batch_codes = encode(
    #         self.coords, batch=self.batch, depth=self.depth, order="z"
    #     )
    #     z_decoded, z_decoded_batch = decode(z_batch_codes, depth=self.depth, order="z")
    #     assert torch.allclose(self.coords, z_decoded), "Z-order批量解码错误"
    #     assert torch.allclose(self.batch, z_decoded_batch), "Z-order批次ID不匹配"

    #     # Hilbert批量测试
    #     h_batch_codes = encode(
    #         self.coords, batch=self.batch, depth=self.depth, order="hilbert"
    #     )
    #     h_decoded, h_decoded_batch = decode(
    #         h_batch_codes, depth=self.depth, order="hilbert"
    #     )
    #     assert torch.allclose(self.coords, h_decoded), "Hilbert批量解码错误"
    #     assert torch.allclose(self.batch, h_decoded_batch), "Hilbert批次ID不匹配"

    #     print("批量处理测试通过!")

    # def test_performance_comparison(self):
    #     """性能对比测试"""
    #     print("\n=== 性能对比测试 ===")

    #     # Z-order性能
    #     z_start = time.time()
    #     z_codes = z_order_encode(self.coords, depth=self.depth)
    #     z_order_decode(z_codes, depth=self.depth)
    #     z_duration = time.time() - z_start

    #     # Hilbert性能
    #     h_start = time.time()
    #     h_codes = hilbert_encode(self.coords, depth=self.depth)
    #     hilbert_decode(h_codes, depth=self.depth)
    #     h_duration = time.time() - h_start

    #     print(f"Z-order总耗时: {z_duration:.4f}秒")
    #     print(f"Hilbert总耗时: {h_duration:.4f}秒")
    #     print(f"性能差异: {abs(z_duration - h_duration):.4f}秒")
