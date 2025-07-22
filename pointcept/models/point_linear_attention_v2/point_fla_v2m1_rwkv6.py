#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @FileName      : point_fla_v2m1_rwkv6
# @Time          : 2025-07-20 22:14:45
# @Author        : TgM
# @Email         : Tgmmmmmmmm@email.swu.edu.cn
# @description   : Prwkv6 v2
"""
# diy rwkv6
from .rwkv6 import RWKV6Attention

# import fla.layers
from fla.layers.gla import GatedLinearAttention
from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.multiscale_retention import MultiScaleRetention

from addict import Dict
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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

        offsets = point.offset  # [B], 每个batch的结束位置+1

        cu_seqlens = torch.cat([torch.tensor([0], device=feat.device), offsets])

        if self.model_type in ["rwkv6", "rwkv7"]:
            output = self._forward_rwkv(feat, cu_seqlens)
        elif self.model_type == "gla":
            output = self._forward_gla(feat, cu_seqlens)
        elif self.model_type == "retnet":
            output = self._forward_retnet(feat, cu_seqlens)

        point.feat = self.proj_drop(output)

        return point

    def _forward_rwkv(
        self, feat: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """处理RWKV系列的特殊参数（如v_first）"""
        kwargs = {}
        if self.model_type == "rwkv7":
            kwargs["v_first"] = None  # 初始化时设为None，首次调用会自动生成
        attn_out, _, _ = self.line_attn(
            hidden_states=feat.unsqueeze(0), cu_seqlens=cu_seqlens, **kwargs
        )
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
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        fla_model="rwkv6",
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

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
        self.ls1 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
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
        self.ls2 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
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
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
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
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
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

        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("Prwkv-v2m1")
class PointFlashLinearAttentionV2(PointModule):
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
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        traceable=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        # assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        # assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
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
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
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
                        layer_scale=layer_scale,
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
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
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
                            layer_scale=layer_scale,
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
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

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
