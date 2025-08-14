import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_
from .diy_order import matrix2key as diy_order_encode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z", rule_matrix=None):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "diy", "diy-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "diy":
        assert rule_matrix is not None
        code = diy_order_encode(grid_coord, rule_matrix, depth=depth)
    elif order == "diy-trans":
        assert rule_matrix is not None
        code = diy_order_encode(grid_coord[:, [1, 0, 2]], rule_matrix, depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z, b = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)


def diy_order_encode(
    grid_coord: torch.Tensor, rule_matrix: torch.Tensor, depth: int = 16
):
    """
    DIY order encoding based on a rule matrix.
    Version 1 , only fixed size, fixed rule_matrix
    """
    # 版本1: 固定每层都相同大小，固定的规则矩阵
    level_list = [1] * depth
    submatrices = [rule_matrix] * depth

    dim = 3

    return diy_order_encode_(
        coords=grid_coord,
        dim=dim,
        depth=depth,
        level_list=level_list,
        submatrices=submatrices,
    )
