"""Level decoder with multi-scale UNet fusion for temporal prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvLSTMCell, downsample_shape, zero_lstm_state


class LevelDecoder(nn.Module):
    """Per-level decoder: binary latent -> reconstructed residual contribution."""

    def __init__(
        self,
        bits: int = 32,
        fuse_context: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_context = fuse_context

        self.conv1 = nn.Conv2d(bits, 512, kernel_size=1, bias=False)
        self.rnn1 = ConvLSTMCell(512, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1)
        self.rnn2 = ConvLSTMCell(128, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1)
        self.rnn3 = ConvLSTMCell(128, 256, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3)
        self.rnn4 = ConvLSTMCell(64, 128, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3)
        self.out = nn.Conv2d(32, 3, kernel_size=1, bias=False)

    def init_state(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        h16, w16 = downsample_shape(height, width, 16)
        h8, w8 = downsample_shape(height, width, 8)
        h4, w4 = downsample_shape(height, width, 4)
        h2, w2 = downsample_shape(height, width, 2)
        return (
            zero_lstm_state(batch, 512, h16, w16, device),
            zero_lstm_state(batch, 512, h8, w8, device),
            zero_lstm_state(batch, 256, h4, w4, device),
            zero_lstm_state(batch, 128, h2, w2, device),
        )

    def forward(
        self,
        symbols: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        unet_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        h1, h2, h3, h4 = state

        x = self.conv1(symbols)
        h1 = self.rnn1(x, h1)

        # Fuse UNet features at matching scales (1/8, 1/4, 1/2)
        x = F.pixel_shuffle(h1[0], 2)
        if self.fuse_context and unet_features is not None:
            x = x + unet_features[0] # Scale 1/8
        h2 = self.rnn2(x, h2)

        x = F.pixel_shuffle(h2[0], 2)
        if self.fuse_context and unet_features is not None:
            x = x + unet_features[1] # Scale 1/4
        h3 = self.rnn3(x, h3)

        x = F.pixel_shuffle(h3[0], 2)
        if self.fuse_context and unet_features is not None:
            x = x + unet_features[2] # Scale 1/2
        h4 = self.rnn4(x, h4)

        x = F.pixel_shuffle(h4[0], 2)
        out = torch.tanh(self.out(x)) / 2.0
        return out, (h1, h2, h3, h4)
