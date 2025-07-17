"""
Octree Transformer

Modified from https://github.com/octree-nn/octformer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# DropPath
from timm.models.layers import DropPath

from torch import Tensor

# Mamba
import math
from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba, Block

# from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from pointcept.models.octmamba.PointMamba import PointMambaBlock

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    import ocnn
    from ocnn.octree import Octree, Points
except ImportError:
    from pointcept.utils.misc import DummyClass

    ocnn = None
    Octree = DummyClass
    Points = DummyClass

try:
    import dwconv
except ImportError:
    dwconv = None

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch


class OctreeDWConvBn(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
    ):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False
        )
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out


class OctreeMamba(nn.Module):

    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        self.pim = PointMambaMix(input_dim=dim, output_dim=dim, fused_add_norm=True)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = data.unsqueeze(0)
        data = self.pim(data)
        data = data.squeeze(0)
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)  # noqa


class OctMambaBlock(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        nempty: bool = True,
        activation: torch.nn.Module = torch.nn.GELU,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mamba = OctreeMamba(dim, proj_drop)
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.mamba(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data


class OctMambaStage(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        nempty: bool = True,
        activation: torch.nn.Module = torch.nn.GELU,
        interval: int = 6,
        use_checkpoint: bool = True,
        num_blocks: int = 2,
        pim_block=PointMambaBlock,
        **kwargs,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_checkpoint = use_checkpoint
        self.interval = interval  # normalization interval
        self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList(
            [
                pim_block(
                    dim=dim,
                    proj_drop=proj_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    nempty=nempty,
                    activation=activation,
                )
                for i in range(num_blocks)
            ]
        )
        # self.norms = torch.nn.ModuleList([
        #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(
                    self.blocks[i], data, octree, depth, use_reentrant=False
                )
            else:
                data = self.blocks[i](data, octree, depth)
            # if i % self.interval == 0 and i != 0:
            #   data = self.norms[(i - 1) // self.interval](data)
        return data


class OctMambaDecoder(torch.nn.Module):
    def __init__(
        self, channels: List[int], fpn_channel: int, nempty: bool, head_up: int = 1
    ):
        super().__init__()
        self.head_up = head_up
        self.num_stages = len(channels)
        self.conv1x1 = torch.nn.ModuleList(
            [
                torch.nn.Linear(channels[i], fpn_channel)
                for i in range(self.num_stages - 1, -1, -1)
            ]
        )
        self.upsample = ocnn.nn.OctreeUpsample("nearest", nempty)
        self.conv3x3 = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    fpn_channel, fpn_channel, kernel_size=[3], stride=1, nempty=nempty
                )
                for _ in range(self.num_stages)
            ]
        )
        self.up_conv = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeDeconvBnRelu(
                    fpn_channel, fpn_channel, kernel_size=[3], stride=2, nempty=nempty
                )
                for _ in range(self.head_up)
            ]
        )

    def forward(self, features: Dict[int, torch.Tensor], octree: Octree):
        depth = min(features.keys())
        depth_max = max(features.keys())
        assert self.num_stages == len(features)

        feature = self.conv1x1[0](features[depth])
        conv_out = self.conv3x3[0](feature, octree, depth)
        out = self.upsample(conv_out, octree, depth, depth_max)
        for i in range(1, self.num_stages):
            depth_i = depth + i
            feature = self.upsample(feature, octree, depth_i - 1)
            feature = self.conv1x1[i](features[depth_i]) + feature
            conv_out = self.conv3x3[i](feature, octree, depth_i)
            out = out + self.upsample(conv_out, octree, depth_i, depth_max)
        for i in range(self.head_up):
            out = self.up_conv[i](out, octree, depth_max + i)
        return out


class PatchEmbed(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 96,
        num_down: int = 2,
        nempty: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    in_channels if i == 0 else channels[i],
                    channels[i],
                    kernel_size=[3],
                    stride=1,
                    nempty=nempty,
                )
                for i in range(self.num_stages)
            ]
        )
        self.downsamples = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    channels[i],
                    channels[i + 1],
                    kernel_size=[2],
                    stride=2,
                    nempty=nempty,
                )
                for i in range(self.num_stages)
            ]
        )
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty
        )

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = (2,),
        nempty: bool = True,
    ):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            nempty=nempty,
            use_bias=True,
        )

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


@MODELS.register_module("OctMamba-v1m1")
class OctMamba(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        fpn_channels=168,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        stem_down=2,
        head_up=2,
        drop_path=0.5,
        nempty=True,
        octree_scale_factor=10.24,
        octree_depth=11,
        octree_full_depth=2,
    ):
        super().__init__()
        assert ocnn is not None, "Please follow `README.md` to install ocnn.`"
        assert dwconv is not None, "Please follow `README.md` to install dwconv.`"

        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        self.octree_scale_factor = octree_scale_factor
        self.octree_depth = octree_depth
        self.octree_full_depth = octree_full_depth
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList(
            [
                OctMambaStage(
                    dim=channels[i],
                    drop_path=drop_ratio[
                        sum(num_blocks[:i]) : sum(num_blocks[: i + 1])
                    ],
                    nempty=nempty,
                    num_blocks=num_blocks[i],
                )
                for i in range(self.num_stages)
            ]
        )
        self.downsamples = torch.nn.ModuleList(
            [
                Downsample(channels[i], channels[i + 1], kernel_size=[2], nempty=nempty)
                for i in range(self.num_stages - 1)
            ]
        )
        self.decoder = OctMambaDecoder(
            channels=channels, fpn_channel=fpn_channels, nempty=nempty, head_up=head_up
        )
        self.interp = ocnn.nn.OctreeInterp("nearest", nempty)
        self.seg_head = (
            nn.Sequential(
                nn.Linear(fpn_channels, fpn_channels),
                torch.nn.BatchNorm1d(fpn_channels),
                nn.ReLU(inplace=True),
                nn.Linear(fpn_channels, num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, data_dict):
        coord = data_dict["coord"]
        normal = data_dict["normal"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        point = Points(
            points=coord / self.octree_scale_factor,
            normals=normal,
            features=feat,
            batch_id=batch.unsqueeze(-1),
            batch_size=len(offset),
        )
        octree = ocnn.octree.Octree(
            depth=self.octree_depth,
            full_depth=self.octree_full_depth,
            batch_size=len(offset),
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        feat = self.patch_embed(octree.features[octree.depth], octree, octree.depth)
        depth = octree.depth - self.stem_down  # current octree depth

        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            feat = self.layers[i](feat, octree, depth_i)
            features[depth_i] = feat
            if i < self.num_stages - 1:
                feat = self.downsamples[i](feat, octree, depth_i)
        out = self.decoder(features, octree)
        # interp representation to points before Octreeization
        query_pts = torch.cat([point.points, point.batch_id], dim=1).contiguous()
        out = self.interp(out, octree, octree.depth, query_pts)
        out = self.seg_head(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (self.drop_path(hidden_states) + residual)
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        **ssm_cfg,
        **factory_kwargs,
    )  # 创建Mamba类的实例

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
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


class PointMambaMix(nn.Module):
    def __init__(
        self,
        output_dim=512,
        input_dim=512,
        drop_path=0.1,
        drop_out_in_block=0.1,
        n_layer=1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        # bimamba_type="none",
        bimamba_type="v2",
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    input_dim,  # 嵌入x
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            input_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_out_in_block = (
            nn.Dropout(drop_out_in_block) if drop_out_in_block > 0.0 else nn.Identity()
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward_features(self, input_ids, inference_params=None):
        # hidden_states = self.embedding(input_ids)
        hidden_states = input_ids

        # print('input_ids.shape',input_ids.shape)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )

            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # offset
        hidden_states = hidden_states - input_ids

        return hidden_states

    def forward(self, input_ids, inference_params=None):
        input_ids = self.forward_features(input_ids, inference_params)
        return input_ids
