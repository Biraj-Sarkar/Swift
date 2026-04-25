"""Multi-level Swift-style autoencoder pipeline with Temporal Warping.

Each level runs:
encoder -> binarizer -> entropy encoder -> entropy decoder -> decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import os
import torch
import torch.nn as nn

from .binarizer import LevelBinarizer
from .decoder import LevelDecoder
from .encoder import LevelEncoder
from .entropy import BitstreamTensor, EntropyDecoder, LearnedEntropyModel
from .common import UNetContext, warp


@dataclass
class LevelOutput:
    latent: torch.Tensor
    symbols: torch.Tensor
    reconstruction_delta: torch.Tensor
    reconstruction: torch.Tensor  # Cumulative reconstruction up to this level
    bitstream: BitstreamTensor
    rate_bpp: torch.Tensor
    entropy_logits: torch.Tensor
    entropy_probabilities: torch.Tensor


class MultiLevelAutoencoder(nn.Module):
    """Stack of coding levels with temporal prediction and residual refinement."""

    def __init__(
        self,
        num_levels: int = 5,
        bits: int = 32,
        fuse_context: bool = True,
    ) -> None:
        super().__init__()

        self.num_levels = num_levels
        self.bits = bits
        self.fuse_context = fuse_context

        # Shared UNet for context extraction
        self.unet = UNetContext(in_channels=6) if fuse_context else None

        self.encoders = nn.ModuleList(
            [
                LevelEncoder(
                    in_channels=3, # Current Frame Residual
                    fuse_context=fuse_context,
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
                LevelDecoder(bits=bits, fuse_context=fuse_context)
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
        ref_frames: tuple[torch.Tensor, torch.Tensor] | None = None,
        motion_vectors: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_entropy_decode: bool | None = None,
    ) -> tuple[torch.Tensor, list[LevelOutput], torch.Tensor]:
        """Forward pass through all coding levels.

        x: Current Frame [N, 3, H, W]
        ref_frames: (Past Frame, Future Frame)
        motion_vectors: (Flow Past-to-Current, Flow Future-to-Current)
        """

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must be NCHW RGB tensor with shape [N, 3, H, W].")

        batch, _, height, width = x.shape
        device = x.device

        # 1. Temporal Prediction and Residual Init
        unet_features = None
        prediction = torch.zeros_like(x) + 0.5

        if self.fuse_context and ref_frames and motion_vectors:
            past, future = ref_frames
            mv_past, mv_future = motion_vectors

            # Warp reference frames to align with current frame
            warped_past = warp(past, mv_past)
            warped_future = warp(future, mv_future)

            # Extract high-level features for the decoder
            combined_context = torch.cat([warped_past, warped_future], dim=1)
            unet_features = self.unet(combined_context)

            # Use average of warped frames as an initial prediction
            prediction = (warped_past + warped_future) / 2.0

        enc_states, dec_states = self.init_states(batch, height, width, device)

        # We encode the difference between the frame and our prediction
        residual = x - prediction
        reconstruction = prediction.clone()

        outputs: list[LevelOutput] = []
        total_rate_bpp = torch.zeros((), device=device)

        if use_entropy_decode is None:
            use_entropy_decode = not self.training

        for level in range(self.num_levels):
            # Encoder focuses on the residual
            latent, enc_states[level] = self.encoders[level](residual, enc_states[level], unet_features=unet_features)

            symbols = self.binarizers[level](latent)

            # Entropy loss estimation (Always active during training)
            entropy = self.entropy_models[level](latent, symbols, image_shape=(height, width))
            bitstream = entropy.bitstream

            # Passing encoded codes: training uses symbols, evaluation can use entropy decode
            decoded_symbols = symbols if not use_entropy_decode else self.entropy_decoders[level](bitstream).to(
                device=device,
                dtype=symbols.dtype,
            )

            # Decoder reconstructs the "changes" (deltas) to the prediction
            delta, dec_states[level] = self.decoders[level](decoded_symbols, dec_states[level], unet_features=unet_features)

            residual = residual - delta
            reconstruction = reconstruction + delta
            total_rate_bpp = total_rate_bpp + entropy.rate_bpp

            outputs.append(
                LevelOutput(
                    latent=latent,
                    symbols=symbols,
                    reconstruction_delta=delta,
                    reconstruction=reconstruction.clone(),
                    bitstream=bitstream,
                    rate_bpp=entropy.rate_bpp,
                    entropy_logits=entropy.logits,
                    entropy_probabilities=entropy.probabilities,
                )
            )

        reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction, outputs, total_rate_bpp

    def save_model(self, directory: str = None, filename: str = "autoencoder.pth") -> str:
        """Saves the model state dict to the project's models directory."""
        if directory is None:
            # Calculate project root (2 levels up from codec/autoencoder/model.py)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            directory = os.path.join(project_root, "models")

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save(self.state_dict(), path)
        print(f"Autoencoder model saved to {path}")
        return path
