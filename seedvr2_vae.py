"""Native SeedVR2 VAE architecture used by the ComfyUI SVDInt4 plugin."""

from __future__ import annotations

from .seedvr2_common import (
    NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND,
    get_logger,
    get_sequence_parallel_world_size,
    retry_on_oom,
    safe_interpolate_operation,
    safe_pad_operation,
)


# ---- models/video_vae_v3/modules/types.py ----
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from enum import Enum
from typing import Dict, Literal, NamedTuple, Optional
import torch

_receptive_field_t = Literal["half", "full"]
_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]
_gradient_checkpointing_t = Optional[Literal["half", "full"]]
_selective_checkpointing_t = Optional[Literal["coarse", "fine"]]

class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.FloatTensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean**2 + self.var - 1.0 - self.logvar,
            dim=list(range(1, self.mean.ndim)),
        )

class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip, need to reset the memory bank.
    State[Active]:          There has been some data in the memory bank.
    State[Unset]:           Error state, indicating users didn't pass correct memory state in.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


class QuantizerOutput(NamedTuple):
    latent: torch.Tensor
    extra_loss: torch.Tensor
    statistics: Dict[str, torch.Tensor]


class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor


# ---- models/video_vae_v3/modules/global_config.py ----
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Optional

_NORM_LIMIT = float("inf")


def get_norm_limit():
    return _NORM_LIMIT


def set_norm_limit(value: Optional[float] = None):
    global _NORM_LIMIT
    if value is None:
        value = float("inf")
    _NORM_LIMIT = value


# ---- models/video_vae_v3/modules/native_layers.py ----
import math
import operator
from types import SimpleNamespace
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderKLOutput(NamedTuple):
    latent_dist: "DiagonalGaussianDistribution"


class DecoderOutput(NamedTuple):
    sample: torch.Tensor


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = torch.zeros_like(self.mean)
            self.std = torch.zeros_like(self.mean)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * sample

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        dims = list(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=dims)
        return 0.5 * torch.sum(
            ((self.mean - other.mean).pow(2) / other.var)
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dims,
        )

    def nll(self, sample: torch.Tensor, dims=None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        if dims is None:
            dims = list(range(1, self.mean.ndim))
        logtwopi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var, dim=dims)

    def mode(self) -> torch.Tensor:
        return self.mean


class SeedVR2AutoencoderKLBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = SimpleNamespace(**kwargs)
        self.use_slicing = False

    @property
    def device(self) -> torch.device:
        for tensor in list(self.parameters(recurse=True)) + list(self.buffers(recurse=True)):
            return tensor.device
        return torch.device("cpu")

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def _convert_deprecated_attention_blocks(self, state_dict):
        return state_dict


class LoRACompatibleConv(nn.Conv2d):
    pass


def _activation(name: str):
    if name in ("swish", "silu"):
        return nn.SiLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.name = name
        self.norm = None
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps=eps or 1e-5, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps=eps or 1e-5, elementwise_affine=elementwise_affine)

        if use_conv:
            self.conv = nn.Conv2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=bias,
            )
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.use_conv and self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding: int = 1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias: bool = True,
        interpolate: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.norm = None
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps=eps or 1e-5, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps=eps or 1e-5, elementwise_affine=elementwise_affine)

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        elif use_conv:
            conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size or 3, padding=padding, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        if self.interpolate:
            hidden_states = F.interpolate(hidden_states, size=output_size, scale_factor=None if output_size else 2.0, mode="nearest")
        if self.use_conv:
            conv = self.conv if self.name == "conv" else self.Conv2d_0
            hidden_states = conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: Optional[int] = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        self.up = up
        self.down = down
        self.use_in_shortcut = (
            self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        )

        groups_out = groups_out or groups
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=self.out_channels, eps=eps, affine=True)
        self.nonlinearity = _activation(non_linearity)
        self.upsample = Upsample2D(in_channels, use_conv=False) if up else None
        self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op") if down else None
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        conv2_out = conv_2d_out_channels or self.out_channels
        self.conv2 = nn.Conv2d(self.out_channels, conv2_out, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        time_out_dim = self.out_channels * 2 if time_embedding_norm == "scale_shift" else self.out_channels
        self.time_emb_proj = nn.Linear(temb_channels, time_out_dim) if temb_channels is not None else None

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv2_out,
                kernel_size=1 if conv_shortcut else 1,
                bias=conv_shortcut_bias,
            )


class DownEncoderBlock2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class UpDecoderBlock2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class SpatialNorm(nn.Module):
    def __init__(self, f_channels: int, zq_channels: int):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1)

    def forward(self, f: torch.Tensor, zq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if zq is None:
            return self.norm_layer(f)
        if zq.ndim == 5:
            zq = zq[:, :, 0]
        if zq.shape[-2:] != f.shape[-2:]:
            zq = F.interpolate(zq, size=f.shape[-2:], mode="nearest")
        norm_f = self.norm_layer(f)
        return norm_f * self.conv_y(zq) + self.conv_b(zq)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        eps: float = 1e-5,
        out_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.norm_cross = False
        self.norm_q = None
        self.norm_k = None
        self.spatial_norm = SpatialNorm(query_dim, spatial_norm_dim) if spatial_norm_dim is not None else None
        self.group_norm = (
            nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
            if norm_num_groups is not None
            else None
        )
        cross_attention_dim = cross_attention_dim or query_dim
        out_dim = out_dim or query_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, out_dim, bias=out_bias), nn.Dropout(dropout)])

    def prepare_attention_mask(self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3):
        if attention_mask is None:
            return None
        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        if out_dim == 3 and attention_mask.shape[0] < batch_size * self.heads:
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1).repeat_interleave(self.heads, dim=1)
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / self.rescale_output_factor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = torch.Size(dim)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if elementwise_affine and bias else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def is_torch_version(op: str, version: str) -> bool:
    def parse(v: str):
        parts = []
        for token in v.split("+", 1)[0].split("."):
            number = ""
            for char in token:
                if char.isdigit():
                    number += char
                else:
                    break
            if number:
                parts.append(int(number))
        return tuple(parts)

    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }
    return ops[op](parse(torch.__version__), parse(version))


def apply_forward_hook(fn):
    return fn


# ---- models/video_vae_v3/modules/context_parallel_lib.py ----
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor


# Single GPU inference - no distributed processing needed
# print("Warning: Using single GPU inference mode - distributed features disabled")


def causal_conv_slice_inputs(x, split_size, memory_state):
    # Single GPU inference - no slicing needed, return full tensor
    return x


def causal_conv_gather_outputs(x):
    # Single GPU inference - no gathering needed, return tensor as is
    return x


def get_output_len(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    return output_len


def get_cache_size(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    remain_len = (
        input_len + pad_len - ((output_len - 1) * conv_module.stride[dim] + dilated_kernerl_size)
    )
    overlap_len = dilated_kernerl_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len  # >= 0

    assert output_len > 0
    return cache_len


def cache_send_recv(tensor: List[Tensor], cache_size, times, memory=None):
    # Single GPU inference - simplified cache handling
    recv_buffer = None

    # Handle memory buffer for single GPU case
    if memory is not None:
        recv_buffer = memory.to(tensor[0])
    elif times > 0:
        tile_repeat = [1] * tensor[0].ndim
        tile_repeat[2] = times
        recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)

    return recv_buffer


# ---- models/video_vae_v3/modules/causal_inflation_lib.py ----
# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import math
from contextlib import contextmanager
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d


# Single GPU inference - no distributed processing needed
#print("Warning: Using single GPU inference mode - distributed features disabled in causal_inflation_lib")

# Mock distributed functions for single GPU inference
def get_sequence_parallel_group():
    return None

def get_sequence_parallel_rank():
    return 0

def get_sequence_parallel_world_size():
    return 1

def get_next_sequence_parallel_rank():
    return 0

def get_prev_sequence_parallel_rank():
    return 0


@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.
        self.memory_limit = float("inf")

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    def _conv_forward(self, input, weight, bias, *args, **kwargs):
        """
        Override _conv_forward to work around NVIDIA Conv3d memory bug.

        Bug: PyTorch 2.9-2.10 with cuDNN >= 91002 uses 3x memory for Conv3d
        with fp16/bfloat16 weights due to buggy dispatch layer.

        Workaround: Call torch.cudnn_convolution directly to bypass buggy layer.
        Status is logged at startup in compatibility.py.
        """
        if (NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND and
            weight.dtype in (torch.float16, torch.bfloat16) and
            hasattr(torch.backends.cudnn, 'is_available') and
            torch.backends.cudnn.is_available() and
            getattr(torch.backends.cudnn, 'enabled', True)):
            try:
                # Direct cuDNN call bypasses buggy PyTorch dispatch layer (NVIDIA only)
                out = torch.cudnn_convolution(
                    input, weight, self.padding, self.stride, self.dilation, self.groups,
                    benchmark=False, deterministic=False, allow_tf32=True
                )
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            except RuntimeError:
                # Fallback if direct cuDNN call fails (dev builds, edge cases)
                pass

        # Use standard path for unaffected configurations or if workaround failed
        return super()._conv_forward(input, weight, bias, *args, **kwargs)

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        # Compatible with no limit.
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return super().forward(x)

        # Compute tensor shape after concat & padding.
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            x_concat = x
            if prev_cache is not None:
                x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)

            def pad_and_forward():
                padded = safe_pad_operation(x_concat, padding, mode='constant', value=0.0)
                with ignore_padding(self):
                    return Conv3d.forward(self, padded)

            return retry_on_oom(
                pad_and_forward,
                debug=getattr(self, 'debug', None),
                operation_name="InflatedCausalConv3d.pad_and_forward"
            )

        # Exceed memory limit, splitting tensor

        # Split input (& prev_cache).
        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        # Loop Fwd.
        cache = None
        for idx in range(len(x)):
            # Concat prev cache from last dim
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            # Get padding pattern.
            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            # Prepare cache for next slice (this dim).
            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)
                )

            # Recursive.
            x[idx] = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache
            )

            # Update cache.
            cache = next_cache

        output = retry_on_oom(
            torch.cat,
            x,
            split_dim,
            debug=getattr(self, 'debug', None),
            operation_name="InflatedCausalConv3d.concat_splits"
        )
        return output

    def forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        if (
            math.isinf(self.memory_limit)
            and torch.is_tensor(input)
            and get_sequence_parallel_group() is None
        ):
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")
        return super().forward(input)

    def slicing_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2
        )

        # Single GPU inference - simplified memory management
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            if cache_size <= input[-1].size(2):
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()
                if self.memory_device == "cpu" and self.memory is not None:
                    self.memory = self.memory.to("cpu")

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

    def tflops(self, args, kwargs, output) -> float:
        if torch.is_tensor(output):
            output_numel = output.numel()
        elif isinstance(output, list):
            output_numel = sum(o.numel() for o in output)
        else:
            raise NotImplementedError
        return (2 * math.prod(self.kernel_size) * self.in_channels * (output_numel / 1e6)) / 1e6

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=inflate_weight,
                inflate_bias_fn=inflate_bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy > get_norm_limit():
                num_chunks = min(4 if x.element_size() == 2 else 2, norm_layer.num_groups)
                assert norm_layer.num_groups % num_chunks == 0
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                x = list(x.chunk(num_chunks, dim=1))
                weights = norm_layer.weight.chunk(num_chunks, dim=0)
                biases = norm_layer.bias.chunk(num_chunks, dim=0)

                for i, (w, b) in enumerate(zip(weights, biases)):
                    def apply_group_norm():
                        return F.group_norm(x[i], num_groups_per_chunk, w, b, norm_layer.eps)

                    x[i] = retry_on_oom(
                        apply_group_norm,
                        debug=getattr(norm_layer, 'debug', None),
                        operation_name=f"GroupNorm.chunk_{i}"
                    )
                    x[i] = x[i]

                x = retry_on_oom(
                    torch.cat,
                    x,
                    dim=1,
                    debug=getattr(norm_layer, 'debug', None),
                    operation_name="GroupNorm.concat_chunks"
                )
            else:
                x = retry_on_oom(
                    norm_layer,
                    x,
                    debug=getattr(norm_layer, 'debug', None),
                    operation_name="GroupNorm.direct"
                )
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x
    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    # Single GPU inference - always process
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        inflation_mode: Placeholder to align `inflate_weight`.
    """
    assert bias_3d.shape == bias_2d.shape
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict


# ---- models/video_vae_v3/modules/attn_video_vae.py ----
# Copyright (c) 2023 HuggingFace Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.


from contextlib import nullcontext
from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


logger = get_logger(__name__)  # pylint: disable=invalid-name

class Upsample3D(Upsample2D):
    """A 3D upsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv if self.name == "conv" else self.Conv2d_0

        assert type(conv) is not nn.ConvTranspose2d
        # Note: lora_layer is not passed into constructor in the original implementation.
        # So we make a simplification.
        conv = init_causal_conv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        assert not self.interpolate
        # [Override] MAGViT v2 implementation
        if not self.interpolate:
            upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
            self.upscale_conv = nn.Conv3d(
                self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
            )
            identity = (
                torch.eye(self.channels)
                .repeat(upscale_ratio, 1)
                .reshape_as(self.upscale_conv.weight)
            )
            self.upscale_conv.weight.data.copy_(identity)
            nn.init.zeros_(self.upscale_conv.bias)

        if self.name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split([split_size, hidden_states.size(2) - split_size], dim=2)
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            def upscale_and_rearrange():
                temp = self.upscale_conv(hidden_states[i])
                return rearrange(
                    temp,
                    "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                    x=self.spatial_ratio,
                    y=self.spatial_ratio,
                    z=self.temporal_ratio,
                )

            hidden_states[i] = retry_on_oom(
                upscale_and_rearrange,
                debug=getattr(self, 'debug', None),
                operation_name="Upsample3D.upscale_conv"
            )

        # [Overridden] For causal temporal conv
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            def apply_conv():
                if self.name == "conv":
                    return self.conv(hidden_states, memory_state=memory_state)
                else:
                    return self.Conv2d_0(hidden_states, memory_state=memory_state)

            hidden_states = retry_on_oom(
                apply_conv,
                debug=getattr(self, 'debug', None),
                operation_name="Upsample3D.conv"
            )

        if not self.slicing:
            return hidden_states
        else:
            return torch.cat(hidden_states, dim=2)


class Downsample3D(Downsample2D):
    """A 3D downsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if type(conv) in [nn.Conv2d, LoRACompatibleConv]:
            # Note: lora_layer is not passed into constructor in the original implementation.
            # So we make a simplification.
            conv = init_causal_conv3d(
                self.channels,
                self.out_channels,
                kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                padding=(
                    1 if self.temporal_down else 0,
                    self.padding if self.spatial_down else 0,
                    self.padding if self.spatial_down else 0,
                ),
                inflation_mode=inflation_mode,
            )
        elif type(conv) is nn.AvgPool2d:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv and self.padding == 0 and self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = safe_pad_operation(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = init_causal_conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = init_causal_conv3d(
            self.out_channels,
            self.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        if self.up:
            assert type(self.upsample) is Upsample2D
            self.upsample = Upsample3D(
                self.in_channels,
                use_conv=False,
                inflation_mode=inflation_mode,
                slicing=slicing,
            )
        elif self.down:
            assert type(self.downsample) is Downsample2D
            self.downsample = Downsample3D(
                self.in_channels,
                use_conv=False,
                padding=1,
                name="op",
                inflation_mode=inflation_mode,
            )

        if self.use_in_shortcut:
            self.conv_shortcut = init_causal_conv3d(
                self.in_channels,
                self.conv_shortcut.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )


    def forward(
        self, input_tensor, temb, memory_state: MemoryState = MemoryState.DISABLED, **kwargs
    ):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)
        hidden_states = retry_on_oom(
            self.nonlinearity,
            hidden_states,
            debug=getattr(self, 'debug', None),
            operation_name="ResnetBlock3D.nonlinearity"
        )

        if self.upsample is not None:
            # Some nearest-neighbor kernels fail with large batch sizes.
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor, memory_state=memory_state)
            hidden_states = self.upsample(hidden_states, memory_state=memory_state)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, memory_state=memory_state)
            hidden_states = self.downsample(hidden_states, memory_state=memory_state)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DownEncoderBlock3D(DownEncoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
            downsample_padding=downsample_padding,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # [Override] Replace module.
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UpDecoderBlock3D(UpDecoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
            temb_channels=temb_channels,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                    slicing=slicing,
                )
            )

            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_upsample:
            # [Override] Replace module & use learnable upsample
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        interpolate=False,
                        inflation_mode=inflation_mode,
                        slicing=slicing,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            # [Override] Replace module.
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups if resnet_time_scale_shift == "default" else None
                        ),
                        spatial_norm_dim=(
                            temb_channels if resnet_time_scale_shift == "spatial" else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, memory_state: MemoryState = MemoryState.DISABLED):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state)

        return hidden_states


class Encoder3D(nn.Module):
    r"""
    [Override] override most logics to support extra condition input and causal conv

    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use.
            Supported names are resolved by this runtime.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            Supported names are resolved by this runtime.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        # [Override] add extra_cond_dim, temporal down num
        temporal_down_num: int = 2,
        extra_cond_dim: int = None,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.extra_cond_dim = extra_cond_dim

        self.conv_extra_cond = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # [Override] to support temporal down block design
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last ones

            assert down_block_type == "DownEncoderBlock3D"

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                # Note: Don't know why set it as 0
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

            def zero_module(module):
                # Zero out the parameters of a module and return it.
                for p in module.parameters():
                    p.detach().zero_()
                return module

            self.conv_extra_cond.append(
                zero_module(
                    nn.Conv3d(extra_cond_dim, output_channel, kernel_size=1, stride=1, padding=0)
                )
                if self.extra_cond_dim is not None and self.extra_cond_dim > 0
                else None
            )

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = init_causal_conv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    def forward(
        self,
        sample: torch.FloatTensor,
        extra_cond=None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, memory_state, use_reentrant=False
                )
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

            # sample = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.mid_block), sample, use_reentrant=False
            # )

        else:
            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = down_block(sample, memory_state=memory_state)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use.
            Supported names are resolved by this runtime.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            Supported names are resolved by this runtime.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        # [Override] add temporal up block
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        #print(f"slicing_up_num: {slicing_up_num}")
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            assert up_block_type == "UpDecoderBlock3D"
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = init_causal_conv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    # Note: Just copy from Decoder.
    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        memory_state,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds, memory_state
                    )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class AutoencoderKL(SeedVR2AutoencoderKLBase):
    """
    Lightweight local AutoencoderKL compatibility shell.
    """

    def __init__(self, attention: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A hacky way to remove attention.
        if not attention and hasattr(self, "encoder") and hasattr(self, "decoder"):
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Keep compatibility with older attention block checkpoint key layouts.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class VideoAutoencoderKL(SeedVR2AutoencoderKLBase):
    """
    SeedVR2 3D VAE with a local AutoencoderKL compatibility shell.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        *args,
        **kwargs,
    ):
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = max(1, slicing_sample_min_size // (2**temporal_scale_num))

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # [Override] make sure it can be normally initialized
            down_block_types=tuple(
                [down_block_type.replace("3D", "2D") for down_block_type in down_block_types]
            ),
            up_block_types=tuple(
                [up_block_type.replace("3D", "2D") for up_block_type in up_block_types]
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            *args,
            **kwargs,
        )

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True,
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512),
               tile_overlap: Tuple[int, int] = (64, 64)) -> AutoencoderKLOutput:
        if tiled:
            h = self.tiled_encode(x, tile_size=tile_size, tile_overlap=tile_overlap)
        else:
            h = self.slicing_encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True,
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512),
               tile_overlap: Tuple[int, int] = (64, 64)) -> Union[DecoderOutput, torch.Tensor]:

        if tiled:
            decoded = self.tiled_decode(z, tile_size=tile_size, tile_overlap=tile_overlap)
        else:
            decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _encode(
        self, x: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        # Only transfer if not already on correct device
        _x = x if x.device == self.device else x.to(self.device)

        _x = causal_conv_slice_inputs(_x, self.slicing_sample_min_size, memory_state=memory_state)
        h = self.encoder(_x, memory_state=memory_state)

        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h

        output = causal_conv_gather_outputs(output)

        # MPS memory leak workaround (pytorch/pytorch#155060)
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        # Only transfer back if needed
        return output if output.device == x.device else output.to(x.device)

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        # Only transfer if not already on correct device
        _z = z if z.device == self.device else z.to(self.device)

        _z = causal_conv_slice_inputs(_z, self.slicing_latent_min_size, memory_state=memory_state)

        if self.post_quant_conv is not None:
            _z = self.post_quant_conv(_z, memory_state=memory_state)

        output = self.decoder(_z, memory_state=memory_state)
        output = causal_conv_gather_outputs(output)

        # MPS memory leak workaround (pytorch/pytorch#155060)
        if self.device.type == 'mps':
            torch.mps.empty_cache()

        # Only transfer back if needed
        return output if output.device == z.device else output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size * sp_size, dim=2)
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(encoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules()
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(decoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules()
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._decode(z)

    def tiled_encode(self, x: torch.Tensor, tile_size: Tuple[int, int] = (512, 512),
                     tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        r"""
        Encodes an input tensor `x` by splitting it into spatial tiles in latent space. Temporal is handled by `slicing_encode`.
        `tile_size` and `tile_overlap` are interpreted in output-space pixels and converted to latent-space.
        """
        # Ensure 5D [B, C, F, H, W]
        if x.ndim != 5:
            x = x.unsqueeze(2)

        b, c, f, H, W = x.shape
        tile_h, tile_w = tile_size

        # Only tile if input resolution requires multiple tiles
        if H <= tile_h and W <= tile_w:
            return self.slicing_encode(x)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled encoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap

        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        H_lat_total = (H + scale_factor - 1) // scale_factor
        W_lat_total = (W + scale_factor - 1) // scale_factor

        result = None
        count = None

        num_tiles = ((max(H_lat_total - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W_lat_total - latent_overlap_w, 1) + stride_w - 1) // stride_w)

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Encoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        # Pre-compute common ramp values
        ramp_cache = {}
        if latent_overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=latent_overlap_h, device=x.device, dtype=x.dtype)
            ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if latent_overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=latent_overlap_w, device=x.device, dtype=x.dtype)
            ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

        tile_id = 0
        for y_lat in range(0, H_lat_total, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H_lat_total)
            for x_lat in range(0, W_lat_total, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W_lat_total)

                # Skip if fully within overlap of previous tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue

                # Map latent tile to output-space crop
                y_out = y_lat * scale_factor
                x_out = x_lat * scale_factor
                y_out_end = min(y_lat_end * scale_factor, H)
                x_out_end = min(x_lat_end * scale_factor, W)

                tile_id += 1

                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'encode_tile_boundaries'):
                    self.debug.encode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })

                tile_sample = x[:, :, :, y_out:y_out_end, x_out:x_out_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Encoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Encoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                encoded_tile = self.slicing_encode(tile_sample)

                # Initialize output size using first encoded tile
                if result is None:
                    b_out, c_out, f_lat, _, _ = encoded_tile.shape

                    # Accumulate on offload device if specified and different, else on inference device
                    device = getattr(self, 'tensor_offload_device', None)
                    if device is None or device == encoded_tile.device:
                        device = encoded_tile.device

                    result = torch.zeros(
                        (b_out, c_out, f_lat, H_lat_total, W_lat_total),
                        device=device,
                        dtype=encoded_tile.dtype,
                    )
                    count = torch.zeros((1, 1, 1, H_lat_total, W_lat_total), device=device, dtype=encoded_tile.dtype)

                eff_h_lat = min(y_lat_end - y_lat, encoded_tile.shape[3], result.shape[3] - y_lat)
                eff_w_lat = min(x_lat_end - x_lat, encoded_tile.shape[4], result.shape[4] - x_lat)

                encoded_tile = encoded_tile[:, :, : result.shape[2], :eff_h_lat, :eff_w_lat]

                # Build faded masks
                ov_h = max(0, min(latent_overlap_h, eff_h_lat - 1))
                ov_w = max(0, min(latent_overlap_w, eff_w_lat - 1))

                weight_h = torch.ones((eff_h_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)
                weight_w = torch.ones((eff_w_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h] = ramp_cache['h'][:ov_h]
                    if y_lat_end < H_lat_total:  # Not bottom edge
                        weight_h[-ov_h:] = 1 - ramp_cache['h'][:ov_h]
                if ov_w > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w] = ramp_cache['w'][:ov_w]
                    if x_lat_end < W_lat_total:  # Not right edge
                        weight_w[-ov_w:] = 1 - ramp_cache['w'][:ov_w]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, eff_h_lat, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, eff_w_lat)
                encoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                # Accumulate (move to result device if different)
                if result.device != encoded_tile.device:
                    encoded_tile = encoded_tile.to(result.device)
                    weight_h_5d = weight_h_5d.to(result.device)
                    weight_w_5d = weight_w_5d.to(result.device)

                result[:, :, : encoded_tile.shape[2], y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat] += encoded_tile
                count[:, :, :, y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat].addcmul_(weight_h_5d, weight_w_5d)

        # Move result back to inference device if needed and normalize
        if result.device != x.device:
            result = result.to(x.device)
            count = count.to(x.device)
        result.div_(count.clamp(min=1e-6))

        if x.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def tiled_decode(self, z: torch.Tensor, tile_size: Tuple[int, int] = (512, 512), tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
        r"""
        Decodes a latent tensor `z` by splitting it into spatial tiles only. Temporal is handled by `slicing_decode`.
        """
        if z.ndim != 5:
            z = z.unsqueeze(2)

        b, c, f, H, W = z.shape

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space for spatial tiling
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap

        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)

        # Only tile if latent resolution requires multiple tiles
        if H <= latent_tile_h and W <= latent_tile_w:
            return self.slicing_decode(z)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled decoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)

        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        # Allocate later using first decoded results
        result = None
        count = None

        num_tiles = ((max(H - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W - latent_overlap_w, 1) + stride_w - 1) // stride_w)

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Decoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        # Pre-compute common ramp values (small memory, big time save)
        ramp_cache = {}
        if overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=overlap_h, device=z.device, dtype=z.dtype)
            ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=overlap_w, device=z.device, dtype=z.dtype)
            ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

        tile_id = 0
        for y_lat in range(0, H, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H)
            for x_lat in range(0, W, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W)

                # Skip if fully within overlap of previous tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue

                tile_id += 1

                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'decode_tile_boundaries'):
                    # Map to output space
                    y_out = y_lat * scale_factor
                    x_out = x_lat * scale_factor
                    y_out_end = y_lat_end * scale_factor
                    x_out_end = x_lat_end * scale_factor
                    self.debug.decode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })

                tile_latent = z[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Decoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Decoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                decoded_tile = self.slicing_decode(tile_latent)

                # Initialize result tensors using actual decoded shapes on first tile
                if result is None:
                    b_out, c_out, out_f_tile, _, _ = decoded_tile.shape
                    output_h = H * scale_factor
                    output_w = W * scale_factor

                    # Accumulate on offload device if specified and different, else on inference device
                    device = getattr(self, 'tensor_offload_device', None)
                    if device is None or device == decoded_tile.device:
                        device = decoded_tile.device

                    result = torch.zeros((b_out, c_out, out_f_tile, output_h, output_w), device=device, dtype=decoded_tile.dtype)
                    count = torch.zeros((1, 1, 1, output_h, output_w), device=device, dtype=decoded_tile.dtype)

                # Corresponding output-space placement
                y_out, y_out_end = y_lat * scale_factor, y_lat_end * scale_factor
                x_out, x_out_end = x_lat * scale_factor, x_lat_end * scale_factor

                h_out = y_out_end - y_out
                w_out = x_out_end - x_out

                # Build faded masks
                ov_h_out = max(0, min(overlap_h, h_out - 1))
                ov_w_out = max(0, min(overlap_w, w_out - 1))

                weight_h = torch.ones((h_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)
                weight_w = torch.ones((w_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h_out > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h_out] = ramp_cache['h'][:ov_h_out]
                    if y_lat_end < H:  # Not bottom edge
                        weight_h[-ov_h_out:] = 1 - ramp_cache['h'][:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w_out] = ramp_cache['w'][:ov_w_out]
                    if x_lat_end < W:  # Not right edge
                        weight_w[-ov_w_out:] = 1 - ramp_cache['w'][:ov_w_out]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, h_out, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, w_out)
                decoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                # Accumulate (move to result device if different)
                if result.device != decoded_tile.device:
                    decoded_tile = decoded_tile.to(result.device)
                    weight_h_5d = weight_h_5d.to(result.device)
                    weight_w_5d = weight_w_5d.to(result.device)

                result[:, :, : decoded_tile.shape[2], y_out:y_out_end, x_out:x_out_end] += decoded_tile
                count[:, :, :, y_out:y_out_end, x_out:x_out_end].addcmul_(weight_h_5d, weight_w_5d)

        # Move result back to inference device if needed and normalize
        if result.device != z.device:
            result = result.to(z.device)
            count = count.to(z.device)
        result.div_(count.clamp(min=1e-6)) # In-place normalize

        if z.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False, assign=False):
        # Keep compatibility with older attention block checkpoint key layouts.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self,
        *args,
        spatial_downsample_factor: int,
        temporal_downsample_factor: int,
        freeze_encoder: bool,
        **kwargs,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def encode(self, x: torch.FloatTensor, return_dict: bool = True,
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512),
               tile_overlap: Tuple[int, int] = (64, 64)) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap).latent_dist
        # Use deterministic mode for tiled encoding to avoid artifacts
        z = p.mode().squeeze(2)
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.Tensor, return_dict: bool = True,
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512),
               tile_overlap: Tuple[int, int] = (64, 64)) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap).sample.squeeze(2)
        return CausalDecoderOutput(x)

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = max(1, split_size // self.temporal_downsample_factor)
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
