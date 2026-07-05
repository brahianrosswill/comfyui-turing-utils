"""Local RoPE helpers for the vendored SeedVR2 runtime."""

from math import pi
from typing import Literal, Optional, Union

import torch
from torch import Tensor, broadcast_tensors, nn

try:
    from torch.amp import autocast as _torch_autocast

    def _no_autocast(fn):
        return _torch_autocast("cuda", enabled=False)(fn)

except ImportError:
    from torch.cuda.amp import autocast as _cuda_autocast

    def _no_autocast(fn):
        return _cuda_autocast(enabled=False)(fn)


def rotate_half(tensor: Tensor) -> Tensor:
    if tensor.shape[-1] % 2 != 0:
        raise ValueError(f"Rotary embedding requires an even feature dimension, got {tensor.shape[-1]}")
    pairs = tensor.view(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1)
    return rotated.flatten(start_dim=-2)


@_no_autocast
def apply_rotary_emb(
    freqs: Tensor,
    tensor: Tensor,
    start_index: int = 0,
    scale: Union[float, Tensor] = 1.0,
    seq_dim: int = -2,
) -> Tensor:
    if tensor.ndim == 3:
        seq_len = tensor.shape[seq_dim]
        freqs = freqs[-seq_len:].to(tensor)
    else:
        freqs = freqs.to(device=tensor.device, dtype=tensor.dtype)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > tensor.shape[-1]:
        raise ValueError(
            f"Feature dimension {tensor.shape[-1]} is too small for rotary dimension {rot_dim}"
        )

    while freqs.ndim < tensor.ndim:
        freqs = freqs.unsqueeze(0)

    left = tensor[..., :start_index]
    middle = tensor[..., start_index:end_index]
    right = tensor[..., end_index:]
    middle = (middle * freqs.cos() * scale) + (rotate_half(middle) * freqs.sin() * scale)
    return torch.cat((left, middle, right), dim=-1)


class RotaryEmbedding(nn.Module):
    """Small RoPE module matching the SeedVR2 call surface."""

    def __init__(
        self,
        dim: int,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta: float = 10000,
        max_freq: float = 10,
        num_freqs: int = 1,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: int = 512,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"RotaryEmbedding dim must be positive, got {dim}")
        if use_xpos:
            raise NotImplementedError("SeedVR2 does not use XPos rotary embeddings")
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be >= 1.0")

        if theta_rescale_factor != 1.0 and dim > 2:
            theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for
        self.cache_if_possible = cache_if_possible
        self.learned_freq = learned_freq
        self.interpolate_factor = interpolate_factor
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        half_dim = dim // 2
        if custom_freqs is not None:
            freqs = custom_freqs.float()
        elif freqs_for == "lang":
            steps = torch.arange(half_dim, dtype=torch.float32)
            freqs = torch.pow(torch.as_tensor(theta, dtype=torch.float32), -2 * steps / dim)
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, half_dim, dtype=torch.float32) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported rotary frequency type: {freqs_for}")

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)
        self.register_buffer("cached_freqs", None, persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len: int, device, dtype, offset: int = 0) -> Tensor:
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def get_axial_freqs(self, *dims: int) -> Tensor:
        if not dims:
            raise ValueError("At least one axis dimension is required")

        axis_freqs = []
        for axis, axis_size in enumerate(dims):
            if axis_size <= 0:
                raise ValueError(f"Axis dimensions must be positive, got {dims}")

            if self.freqs_for == "pixel":
                positions = torch.linspace(-1, 1, steps=axis_size, device=self.device)
            else:
                positions = torch.arange(axis_size, device=self.device)

            freqs = self.forward(positions, seq_len=axis_size)
            view_shape = [1] * len(dims)
            view_shape[axis] = axis_size
            axis_freqs.append(freqs.reshape(*view_shape, freqs.shape[-1]))

        broadcasted = broadcast_tensors(*axis_freqs)
        return torch.cat(broadcasted, dim=-1)

    @_no_autocast
    def forward(self, positions: Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tensor:
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and seq_len is not None
            and self.freqs_for != "pixel"
        )

        if (
            should_cache
            and self.cached_freqs is not None
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset : offset + seq_len].detach()

        phases = positions.to(dtype=self.freqs.dtype).unsqueeze(-1) * self.freqs
        freqs = torch.stack((phases, phases), dim=-1).flatten(start_dim=-2)

        if should_cache:
            self.cached_freqs = freqs.detach()

        return freqs
