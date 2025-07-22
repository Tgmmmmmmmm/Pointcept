#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : point_fla_v1m1_base
# @Time          : 2025-07-18 08:26:44
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : import fla
"""
# diy rwkv6
from .rwkv6 import RWKV6Attention

# import fla.layers
from fla.layers.gla import GatedLinearAttention
from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.multiscale_retention import MultiScaleRetention

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential, is_ocnn_module


def check_nan(tensor, name, input=None):
    """检查张量中是否有NaN或Inf值"""
    exist_error = False
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        exist_error = True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        exist_error = True
    if exist_error:
        print_tensor_stats(tensor=tensor, name=name)
        if input is not None:
            print_tensor_stats(tensor=input, name=f"{name}'s input")
    return exist_error


def print_tensor_stats(tensor, name=""):
    print("*" * 20 + f"{name} stats " + "*" * 20)
    print(f"shape: {tensor.shape}")
    print(f"dtype: {tensor.dtype}")
    print(f"mean: {tensor.mean().item():.6f}")
    print(f"std: {tensor.std().item():.6f}")
    print(f"min: {tensor.min().item():.6f}")
    print(f"max: {tensor.max().item():.6f}")
    print(f"abs_max: {tensor.abs().max().item():.6f}")
    print("=" * 65)


class FlashLinearAttentionWrapper(PointModule):
    def __init__(
        self,
        hidden_size,
        num_heads=4,
        expand_k=1.0,
        expand_v=1.0,
        proj_drop=0.0,
        model="rwkv6",
        **kwargs,
    ):
        super().__init__()
        self.model_type = model.lower()
        # 根据model参数初始化对应的注意力模块
        if self.model_type == "rwkv6":
            self.line_attn = RWKV6Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                **kwargs,
            )
        elif self.model_type == "rwkv7":
            self.line_attn = RWKV7Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                **kwargs,
            )
        elif self.model_type == "gla":
            self.line_attn = GatedLinearAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                **kwargs,
            )
        elif self.model_type == "retnet":
            self.line_attn = MultiScaleRetention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported model type: {model}. "
                "Available options: 'rwkv6', 'rwkv7', 'gla', 'retnet'"
            )

        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, point: Point):

        feat = point.feat  # [N, C]
        check_nan(feat, "input feat")
        offsets = point.offset  # [B], 每个batch的结束位置+1
        check_nan(offsets, "attn offsets")
        cu_seqlens = torch.cat([torch.tensor([0], device=feat.device), offsets])
        check_nan(cu_seqlens, "attn cu_seqlens")

        if self.model_type in ["rwkv6", "rwkv7"]:
            output = self._forward_rwkv(feat, cu_seqlens)
        elif self.model_type == "gla":
            output = self._forward_gla(feat, cu_seqlens)
        elif self.model_type == "retnet":
            output = self._forward_retnet(feat, cu_seqlens)

        check_nan(output, "attn forward model", feat)
        point.feat = self.proj_drop(output)
        check_nan(point.feat, "attn forward proj_drop", output)
        return point

    def _forward_rwkv(
        self, feat: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """处理RWKV系列的特殊参数（如v_first）"""
        kwargs = {}
        if self.model_type == "rwkv7":
            kwargs["v_first"] = None  # 初始化时设为None，首次调用会自动生成
        check_nan(feat, "_forward_rwkv feat")
        attn_out, _, _ = self.line_attn(
            hidden_states=feat.unsqueeze(0), cu_seqlens=cu_seqlens, **kwargs
        )
        check_nan(attn_out, "_forward_rwkv out", feat)
        return attn_out.squeeze(0)

    def _forward_gla(
        self, feat: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """GLA需要处理可能的attention_mask"""
        attn_out, _, _ = self.line_attn(
            hidden_states=feat.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            attention_mask=None,  # 可根据实际需求生成mask
        )
        return attn_out.squeeze(0)

    def _forward_retnet(
        self, feat: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """RetNet需要处理past_key_values"""
        attn_out, _, _ = self.line_attn(
            hidden_states=feat.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            past_key_values=None,  # 推理时可传入缓存
        )
        return attn_out.squeeze(0)


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


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        # patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,  # 保留但不使用
        qk_scale=None,  # 保留但不使用
        attn_drop=0.0,  # 保留但不使用
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,  # 保留但不使用
        enable_flash=True,  # 保留但不使用
        upcast_attention=True,  # 保留但不使用
        upcast_softmax=True,  # 保留但不使用
        fla_model="rwkv6",
        debug=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.cpe_indice_key = cpe_indice_key
        self.debug = debug
        self.cpe = PointSequential(
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

        self.attn = PointSequential(
            FlashLinearAttentionWrapper(
                hidden_size=channels,
                num_heads=num_heads,
                expand_k=1.0,  # 可调整
                expand_v=1.0,  # 可调整
                proj_drop=proj_drop,
                model=fla_model,
            )
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

    def forward(self, point: Point):
        # 检查输入特征
        check_nan(point.feat, f"{self.debug} input feat")
        shortcut = point.feat

        point = self.cpe(point)
        # 检查CPE输出
        check_nan(point.feat, f"{self.debug} cpe", shortcut)

        point.feat = shortcut + point.feat
        # 检查残差连接后
        check_nan(point.feat, f"{self.debug} cpe residual")

        shortcut = point.feat

        if self.pre_norm:
            point = self.norm1(point)
            # 检查norm1输出
            check_nan(point.feat, f"{self.debug} norm1", shortcut)
        temp_check = point.feat
        point = self.drop_path(self.attn(point))
        # 检查attention输出

        check_nan(point.feat, f"{self.debug} attn", temp_check)

        point.feat = shortcut + point.feat
        # 检查attention残差连接后
        check_nan(point.feat, f"{self.debug} attn residual")
        temp_check = point.feat
        if not self.pre_norm:
            point = self.norm1(point)
            # 检查norm1输出
            check_nan(point.feat, f"{self.debug} norm1 (post-norm)", temp_check)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
            # 检查norm2输出
            check_nan(point.feat, f"{self.debug} norm2", shortcut)
        temp_check = point.feat
        point = self.drop_path(self.mlp(point))
        # 检查MLP输出
        check_nan(point.feat, f"{self.debug} mlp", temp_check)

        point.feat = shortcut + point.feat
        # 检查MLP残差连接后
        check_nan(point.feat, f"{self.debug} mlp residual")
        temp_check = point.feat
        if not self.pre_norm:
            point = self.norm2(point)
            # 检查norm2输出
            check_nan(point.feat, f"{self.debug} norm2 (post-norm)", temp_check)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        # 检查最终输出
        if check_nan(point.feat, "final output"):
            raise ValueError(f"出现了NaN值, 问题发生在{self.debug}")

        return point


class SerializedPooling(PointModule):
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


class SerializedUnpooling(PointModule):
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


@MODELS.register_module("Prwkv-v1m1")
class PointFlashLinearAttention(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        # enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        # dec_patch_size=(48, 48, 48, 48),
        fla_model="rwkv6",
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
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
        self.fla_model = fla_model

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        # assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        # assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

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
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
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
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        # patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        fla_model=fla_model,
                        debug=f"Encoder stage{s}",
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
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
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
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            # patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            fla_model=fla_model,
                            debug=f"Decoder stage{s}",
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
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
