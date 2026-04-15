"""Binarizer module for each coding level."""

from __future__ import annotations

import torch
import torch.nn as nn


class LevelBinarizer(nn.Module):
    """Maps latent features to binary symbols in {-1, +1}."""

    def __init__(self, in_channels: int = 512, bits: int = 32) -> None:
        super().__init__()
        self.bits = bits
        self.proj = nn.Conv2d(in_channels, bits, kernel_size=1, bias=False)

    def _stochastic_sign(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-through binarization with Bernoulli-style noise.
        p = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        sample = torch.bernoulli(p)
        hard = sample * 2.0 - 1.0
        return x + (hard - x).detach()

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        if training is None:
            training = self.training

        latent = torch.tanh(self.proj(x))
        if training:
            return self._stochastic_sign(latent)
        return torch.sign(latent)