"""Multi-level Swift-style autoencoder pipeline.

Each level runs:
encoder -> binarizer -> entropy encoder -> entropy decoder -> decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .binarizer import LevelBinarizer
from .decoder import LevelDecoder
from .encoder import LevelEncoder
from .entropy import BitstreamTensor, EntropyDecoder, LearnedEntropyModel


@dataclass
class LevelOutput:
    latent: torch.Tensor
    symbols: torch.Tensor
    reconstruction_delta: torch.Tensor
    bitstream: BitstreamTensor
    rate_bpp: torch.Tensor
    entropy_logits: torch.Tensor
    entropy_probabilities: torch.Tensor


class MultiLevelAutoencoder(nn.Module):
    """Stack of coding levels with residual refinement."""

    def __init__(
        self,
        num_levels: int = 3,
        bits: int = 32,
        stack_context: bool = False,
        fuse_context: bool = False,
        fuse_level: int = 3,
    ) -> None:
        super().__init__()

        self.num_levels = num_levels
        self.bits = bits

        self.encoders = nn.ModuleList(
            [
                LevelEncoder(
                    stack_context=stack_context,
                    fuse_context=fuse_context,
                    fuse_level=fuse_level,
                )
                for _ in range(num_levels)
            ]
        )
        self.binarizers = nn.ModuleList([LevelBinarizer(in_channels=512, bits=bits) for _ in range(num_levels)])
        self.entropy_decoders = nn.ModuleList([EntropyDecoder() for _ in range(num_levels)])
        self.entropy_models = nn.ModuleList(
            [LearnedEntropyModel(latent_channels=512, bits=bits) for _ in range(num_levels)]
        )
        self.decoders = nn.ModuleList(
            [
                LevelDecoder(bits=bits, fuse_context=fuse_context, fuse_level=fuse_level)
                for _ in range(num_levels)
            ]
        )

    def init_states(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[list[tuple[tuple[torch.Tensor, torch.Tensor], ...]], list[tuple[tuple[torch.Tensor, torch.Tensor], ...]]]:
        enc_states = [enc.init_state(batch, height, width, device) for enc in self.encoders]
        dec_states = [dec.init_state(batch, height, width, device) for dec in self.decoders]
        return enc_states, dec_states

    def forward(
        self,
        x: torch.Tensor,
        context: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
        use_entropy_decode: bool | None = None,
    ) -> tuple[torch.Tensor, list[LevelOutput], torch.Tensor]:
        """Forward pass through all coding levels.

        x expects range [0, 1]. Reconstruction is returned in [0, 1].
        """

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must be NCHW RGB tensor with shape [N, 3, H, W].")

        batch, _, height, width = x.shape
        device = x.device

        enc_states, dec_states = self.init_states(batch, height, width, device)

        residual = x - 0.5
        reconstruction = torch.zeros_like(x) + 0.5
        outputs: list[LevelOutput] = []
        total_rate_bpp = torch.zeros((), device=device)

        if use_entropy_decode is None:
            use_entropy_decode = not self.training

        for level in range(self.num_levels):
            latent, enc_states[level] = self.encoders[level](residual, enc_states[level], context=context)

            symbols = self.binarizers[level](latent)

            entropy = self.entropy_models[level](latent, symbols, image_shape=(height, width))
            bitstream = entropy.bitstream
            decoded_symbols = symbols if not use_entropy_decode else self.entropy_decoders[level](bitstream).to(
                device=device,
                dtype=symbols.dtype,
            )

            delta, dec_states[level] = self.decoders[level](decoded_symbols, dec_states[level], context=context)

            residual = residual - delta
            reconstruction = reconstruction + delta
            total_rate_bpp = total_rate_bpp + entropy.rate_bpp

            outputs.append(
                LevelOutput(
                    latent=latent,
                    symbols=symbols,
                    reconstruction_delta=delta,
                    bitstream=bitstream,
                    rate_bpp=entropy.rate_bpp,
                    entropy_logits=entropy.logits,
                    entropy_probabilities=entropy.probabilities,
                )
            )

        reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction, outputs, total_rate_bpp