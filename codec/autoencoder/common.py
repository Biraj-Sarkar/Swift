"""Shared recurrent blocks and shape helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell matching the style used in existing codec modules."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        hidden_kernel_size: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        hidden_kernel_size = _pair(hidden_kernel_size)
        hidden_padding = _pair(hidden_kernel_size[0] // 2)

        gate_channels = 4 * hidden_channels
        self.conv_ih = nn.Conv2d(
            input_channels,
            gate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv_hh = nn.Conv2d(
            hidden_channels,
            gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = hidden
        gates = self.conv_ih(x) + self.conv_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


def zero_lstm_state(
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.zeros(batch, channels, height, width, device=device)
    c = torch.zeros(batch, channels, height, width, device=device)
    return h, c


def downsample_shape(height: int, width: int, factor: int) -> tuple[int, int]:
    return max(1, height // factor), max(1, width // factor)


def fuse_add(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    proj: nn.Module,
) -> torch.Tensor:
    # Project concatenated context to current feature width then add.
    ctx = torch.cat([left, right], dim=1)
    ctx = F.interpolate(ctx, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return x + proj(ctx)