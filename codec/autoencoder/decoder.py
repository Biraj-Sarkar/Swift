"""Level decoder reused from icodec-style recurrent decoder with laplacian-style context fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvLSTMCell, downsample_shape, fuse_add, zero_lstm_state


class LevelDecoder(nn.Module):
    """Per-level decoder: binary latent -> reconstructed residual contribution."""

    def __init__(
        self,
        bits: int = 32,
        fuse_context: bool = False,
        fuse_level: int = 3,
    ) -> None:
        super().__init__()
        self.fuse_context = fuse_context
        self.fuse_level = fuse_level

        self.conv1 = nn.Conv2d(bits, 512, kernel_size=1, bias=False)
        self.rnn1 = ConvLSTMCell(512, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1)
        self.rnn2 = ConvLSTMCell(128, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1)
        self.rnn3 = ConvLSTMCell(128, 256, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3)
        self.rnn4 = ConvLSTMCell(64, 128, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3)
        self.out = nn.Conv2d(32, 3, kernel_size=1, bias=False)

        self.ctx_proj2 = nn.LazyConv2d(128, kernel_size=1, bias=False)
        self.ctx_proj3 = nn.LazyConv2d(128, kernel_size=1, bias=False)
        self.ctx_proj4 = nn.LazyConv2d(64, kernel_size=1, bias=False)

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
        context: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        h1, h2, h3, h4 = state

        x = self.conv1(symbols)
        h1 = self.rnn1(x, h1)

        x = F.pixel_shuffle(h1[0], 2)
        if self.fuse_context and context is not None and self.fuse_level >= 3:
            x = fuse_add(x, context[0][-3], context[1][-3], self.ctx_proj2)
        h2 = self.rnn2(x, h2)

        x = F.pixel_shuffle(h2[0], 2)
        if self.fuse_context and context is not None and self.fuse_level >= 2:
            x = fuse_add(x, context[0][-2], context[1][-2], self.ctx_proj3)
        h3 = self.rnn3(x, h3)

        x = F.pixel_shuffle(h3[0], 2)
        if self.fuse_context and context is not None and self.fuse_level >= 1:
            x = fuse_add(x, context[0][-1], context[1][-1], self.ctx_proj4)
        h4 = self.rnn4(x, h4)

        x = F.pixel_shuffle(h4[0], 2)
        out = torch.tanh(self.out(x)) / 2.0
        return out, (h1, h2, h3, h4)
