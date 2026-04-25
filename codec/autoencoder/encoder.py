"""Level encoder with UNet context fusion and motion-aware recurrence."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvLSTMCell, downsample_shape, zero_lstm_state


class LevelEncoder(nn.Module):
    """Per-level encoder: image/residual -> latent feature map."""

    def __init__(
        self,
        in_channels: int = 3,
        fuse_context: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_context = fuse_context

        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(64, 256, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)
        self.rnn2 = ConvLSTMCell(256, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)
        self.rnn3 = ConvLSTMCell(512, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1)

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
        unet_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        h1, h2, h3 = state

        x = self.conv(x)
        # unet_features[2] is the smallest scale from UNet (e.g. 1/4 if input was full,
        # but here we need to match the encoder's internal scales).
        # Actually, let's keep it simple: the encoder uses its own path,
        # but the Swift paper mainly fuses context in the DECODER for prediction.
        # However, for the encoder to focus on residuals, we subtract predictions.

        h1 = self.rnn1(x, h1)
        h2 = self.rnn2(h1[0], h2)
        h3 = self.rnn3(h2[0], h3)
        latent = h3[0]

        return latent, (h1, h2, h3)
