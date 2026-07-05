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
