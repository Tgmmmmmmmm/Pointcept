#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : pamba_v1m1_base
# @Time          : 2025-05-29 14:06:44
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : Pamba: Enhancing Global Interaction in Point Clouds via State Space Model
"""
from functools import partial
from typing import Optional
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath
import copy
import time
import inspect
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA

try:
    from mamba_ssm.modules.mamba_simple import Block  # Legacy mambav1 file structure
except ImportError:
    from mamba_ssm.modules.block import Block  # mambav2 file structure

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


def create_block(
    d_model,
    ssm_cfg=None,
    d_intermediate=0,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=True,
    bidirectional_strategy="add",
    bidirectional_weight_tie=True,
    device=None,
    dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "BiMamba")
        if ssm_layer not in ["Mamba1", "Mamba2", "BiMamba"]:
            raise ValueError(
                f"Invalid ssm_layer: {ssm_layer}, only support Mamba1, Mamba2 and BiMamba"
            )
        if ssm_layer == "Mamba1":
            mixer_cls = partial(
                Mamba,
                layer_idx=layer_idx,
                **ssm_cfg,
                **bidirectional_kwargs,
                **factory_kwargs,
            )
        elif ssm_layer == "Mamba2":
            mixer_cls = partial(
                Mamba2,
                layer_idx=layer_idx,
                **ssm_cfg,
                **bidirectional_kwargs,
                **factory_kwargs,
            )
        elif ssm_layer == "BiMamba":
            mixer_cls = partial(
                BiMambaWrapper,
                layer_idx=layer_idx,
                **ssm_cfg,
                **bidirectional_kwargs,
                **factory_kwargs,
            )
    else:
        mixer_cls = partial(
            MHA,
            layer_idx=layer_idx,
            **attn_cfg,
            **bidirectional_kwargs,
            **factory_kwargs,
        )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    # mambav2 compatibility
    if "mlp_cls" in inspect.signature(Block.__init__).parameters:
        block = Block(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:

        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:

                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        bidirectional_weight_tie: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!"
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
            if (
                bidirectional_weight_tie
            ):  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """

        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(
                    dims=(1,)
                ),  # Flip along the sequence length dimension
                inference_params=inference_params,
            ).flip(
                dims=(1,)
            )  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(
                    f"`{self.bidirectional_strategy}` for bi-directionality not implemented!"
                )
        return out


class BiMambaMix(PointModule):
    def __init__(
        self,
        d_model: int = 512,
        n_layer=1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        order_index=0,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        bidirectional_weight_tie: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.d_model = d_model
        self.order_index = order_index

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,  # 嵌入x
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bidirectional=bidirectional,
                    bidirectional_strategy=bidirectional_strategy,
                    bidirectional_weight_tie=bidirectional_weight_tie,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, point, inference_params=None, **mixer_kwargs):
        # print("Adaptive Batch Size:", point.batch[-1].item() + 1)
        order = point.serialized_order[self.order_index]
        inverse = point.serialized_inverse[self.order_index]
        feat_ordered = point.feat[order]

        offset = point.offset
        offsets_with_zero = torch.cat([torch.tensor([0], device=offset.device), offset])
        lengths = offsets_with_zero[1:] - offsets_with_zero[:-1]
        # print("lengths:", lengths)
        max_seqlen = lengths.max().item()

        batches = torch.split(feat_ordered, lengths.tolist(), dim=0)
        # 填充到最大长度
        padded_batches = []
        for batch in batches:
            # 计算需要填充的长度
            pad_len = max_seqlen - batch.size(0)
            if pad_len > 0:
                # 在序列末尾填充 0（或其他填充值）
                padded_batch = torch.nn.functional.pad(
                    batch,
                    (
                        0,
                        0,
                        0,
                        pad_len,
                    ),  # (left, right, top, bottom) -> (dim0, dim1, dim2, dim3)
                    mode="constant",
                    value=0,
                )
            else:
                padded_batch = batch
            padded_batches.append(padded_batch.unsqueeze(0))
        # print("padded_batches shape:", [b.shape for b in padded_batches])
        # 转换为 (batch, seqlen, dim)
        batches = torch.cat(padded_batches, dim=0)  # (batch, seqlen, D)
        # print("填充数据", batches.shape)
        hidden_states = batches

        residual = None
        # 执行每一层
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )
        # 如果不进行混合归一化
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        B, MaxL, D = hidden_states.shape
        # 出来的形状跟输入的形状一样
        # print("After Mamba hidden_states shape:", hidden_states.shape)
        hidden_states_flat = hidden_states.reshape(B * MaxL, D)
        # print("hidden_states_flat shape:", hidden_states_flat.shape)
        # print("lengths device:", lengths.device)
        device = lengths.device
        mask = torch.arange(MaxL, device=device).expand(B, MaxL) < lengths.unsqueeze(
            1
        )  # (B, MaxL)
        # print("mask shape:", mask.shape)
        mask_flat = mask.reshape(-1)  # (B * MaxL)
        hidden_states_valid = hidden_states_flat[mask_flat]  # (total_L, D)

        # print("hidden_states_valid shape:", hidden_states_valid.shape)
        # hidden_states (B,MaxL,D) -> remove padding -> B,L,D -> cat -> L,D

        point.feat = hidden_states_valid[inverse]
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMambaBlock(PointSequential):
    def __init__(
        self,
        channels,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=False,
        order_index=0,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        cpe_indice_key=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.spconv1 = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.spconv2 = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.bimamba = BiMambaMix(
            order_index=order_index,
            d_model=channels,
            bidirectional=bidirectional,
            bidirectional_strategy=bidirectional_strategy,
            bidirectional_weight_tie=bidirectional_weight_tie,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward_(self, point: Point):

        # Local aggregation
        shortcut = point.feat
        point = self.spconv1(point)
        point = self.spconv2(point)
        point.feat = shortcut + point.feat

        # Global aggregation
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.bimamba(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class DownSample(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=False,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class UpSample(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("Pamba-v1m1")
class Pamba(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=False,
        cls_mode=False,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        # s -> [0, 1, 2, 3, 4]
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    # Pamba DownSample
                    DownSample(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    # Pamba Block
                    ConvMambaBlock(
                        channels=enc_channels[s],
                        mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        bidirectional=bidirectional,
                        bidirectional_strategy=bidirectional_strategy,
                        bidirectional_weight_tie=bidirectional_weight_tie,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            # s -> [3, 2, 1, 0] (reverse order)
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    # Pamba UpSample
                    UpSample(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        # Pamba ConvMamba
                        ConvMambaBlock(
                            channels=dec_channels[s],
                            mlp_ratio=mlp_ratio,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            bidirectional=bidirectional,
                            bidirectional_strategy=bidirectional_strategy,
                            bidirectional_weight_tie=bidirectional_weight_tie,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):

        # start_time = time.perf_counter()
        point = Point(data_dict)
        # end_time = time.perf_counter()

        # # 计算耗时
        # elapsed_time = end_time - start_time
        # print(f"Point cloud init took {elapsed_time:.6f} seconds")

        # print("=" * 20 + " Point Info " + "=" * 20)
        # print(point.keys())
        # print("coord shape", point.coord.shape, point.coord[0])
        # print("grid_coord shape", point.grid_coord.shape, point.grid_coord[0])
        # print("segment shape", point.segment.shape, point.segment[0])
        # print("feat shape", point.feat.shape, point.feat[0])
        # print("batch shape", point.batch.shape, point.batch[0])
        # print("offset shape", point.offset.shape, point.offset[0])

        # start_time = time.perf_counter()
        point.serialization(order=self.order)
        # end_time = time.perf_counter()

        # # 计算耗时
        # elapsed_time = end_time - start_time
        # print(f"Point cloud serialization took {elapsed_time:.6f} seconds")

        # print("=" * 20 + " After Serialization Info " + "=" * 20)
        # print(point.keys())
        # print(
        #     "serialized_code shape",
        #     point.serialized_code.shape,
        #     point.serialized_code[0][0],
        # )
        # print(
        #     "serialized_order shape",
        #     point.serialized_order.shape,
        #     point.serialized_order[0][0],
        # )
        # print(
        #     "serialized_inverse shape",
        #     point.serialized_inverse.shape,
        #     point.serialized_inverse[0][0],
        # )
        # print("serialized_depth", point.serialized_depth)

        point.sparsify()
        # print("=" * 20 + " After Sparsify Info " + "=" * 20)
        # print(point.keys())
        # print("sparse_shape", point.sparse_shape)

        point = self.embedding(point)
        # print("=" * 20 + " After Embedding Info " + "=" * 20)
        # print(point.keys())
        # print("feat shape", point.feat.shape, point.feat[0])

        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point
