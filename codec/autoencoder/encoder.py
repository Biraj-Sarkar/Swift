"""Level encoder reused from icodec-style recurrent encoder with laplacian-style context fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvLSTMCell, downsample_shape, fuse_add, zero_lstm_state


class LevelEncoder(nn.Module):
    """Per-level encoder: image/residual -> latent feature map."""

    def __init__(
        self,
        stack_context: bool = False,
        fuse_context: bool = False,
        fuse_level: int = 3,
    ) -> None:
        super().__init__()
        self.stack_context = stack_context
        self.fuse_context = fuse_context
        self.fuse_level = fuse_level

        in_channels = 9 if stack_context else 3
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(64, 256, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)
        self.rnn2 = ConvLSTMCell(256, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)
        self.rnn3 = ConvLSTMCell(512, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)

        self.ctx_proj1 = nn.LazyConv2d(64, kernel_size=1, bias=False)
        self.ctx_proj2 = nn.LazyConv2d(256, kernel_size=1, bias=False)
        self.ctx_proj3 = nn.LazyConv2d(512, kernel_size=1, bias=False)

    def init_state(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        h4, w4 = downsample_shape(height, width, 4)
        h8, w8 = downsample_shape(height, width, 8)
        h16, w16 = downsample_shape(height, width, 16)
        return (
            zero_lstm_state(batch, 256, h4, w4, device),
            zero_lstm_state(batch, 512, h8, w8, device),
            zero_lstm_state(batch, 512, h16, w16, device),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        context: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        h1, h2, h3 = state

        x = self.conv(x)
        if self.fuse_context and context is not None and self.fuse_level >= 1:
            x = fuse_add(x, context[0][-1], context[1][-1], self.ctx_proj1)

        h1 = self.rnn1(x, h1)
        x = h1[0]
        if self.fuse_context and context is not None and self.fuse_level >= 2:
            x = fuse_add(x, context[0][-2], context[1][-2], self.ctx_proj2)

        h2 = self.rnn2(x, h2)
        x = h2[0]
        if self.fuse_context and context is not None and self.fuse_level >= 3:
            x = fuse_add(x, context[0][-3], context[1][-3], self.ctx_proj3)

        h3 = self.rnn3(x, h3)
        latent = h3[0]

        return latent, (h1, h2, h3)
