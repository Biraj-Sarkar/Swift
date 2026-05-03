"""Shared recurrent blocks, shape helpers, and temporal warping logic."""

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

def warp(x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """
    Warp an image or feature map with optical flow.
    x: [B, C, H, W]
    flo: [B, 2, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

class UNetContext(nn.Module):
    """
    UNet for extracting features from warped reference frames.
    Returns features at [1/8, 1/4, 1/2] scales to match Decoder's spatial hierarchy.
    """
    def __init__(self, in_channels=6):
        super().__init__()
        # Encoder: Down to 1/8
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 1/2
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 1/4
        self.enc4 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # 1/8

        # Decoder: Up to 1/2
        self.dec1 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1) # 1/4
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 1/2

        # Final mapping to match LevelDecoder/SwiftDecoder channel counts
        # Target channels: 1/8: 128, 1/4: 128, 1/2: 64
        self.map_1_8 = nn.Conv2d(512, 128, 1)
        self.map_1_4 = nn.Conv2d(128, 128, 1)
        self.map_1_2 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))

        d1 = F.relu(self.dec1(e4))
        d2 = F.relu(self.dec2(d1))

        # Returns features at [1/8, 1/4, 1/2] scales
        return [
            self.map_1_8(e4),
            self.map_1_4(d1),
            self.map_1_2(d2)
        ]

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
