"""Entropy encoder/decoder utilities for binary latent codes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_FULL_RANGE = 1 << 32
_HALF = _FULL_RANGE >> 1
_QUARTER = _HALF >> 1
_THREE_QUARTER = _QUARTER * 3


class _BitWriter:
    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, bit: int) -> None:
        self._bits.append(1 if bit else 0)

    def to_bytes(self) -> bytes:
        if not self._bits:
            return b""
        packed = np.packbits(np.array(self._bits, dtype=np.uint8), bitorder="big")
        return packed.tobytes()


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._byte_idx = 0
        self._bit_idx = 0

    def read(self) -> int:
        if self._byte_idx >= len(self._payload):
            return 0
        byte = self._payload[self._byte_idx]
        bit = (byte >> (7 - self._bit_idx)) & 1
        self._bit_idx += 1
        if self._bit_idx == 8:
            self._bit_idx = 0
            self._byte_idx += 1
        return bit


def _binary_arithmetic_encode(symbols: np.ndarray, p0_freqs: np.ndarray, total_freq: int) -> bytes:
    if symbols.size == 0:
        return b""

    low = 0
    high = _FULL_RANGE - 1
    pending = 0
    writer = _BitWriter()

    def emit(bit: int, pending_bits: int) -> int:
        writer.write(bit)
        for _ in range(pending_bits):
            writer.write(1 - bit)
        return 0

    for bit, p0 in zip(symbols.tolist(), p0_freqs.tolist()):
        p0 = int(p0)
        rng = high - low + 1

        split = low + (rng * p0 // total_freq)
        if bit == 0:
            high = split - 1
        else:
            low = split

        while True:
            if high < _HALF:
                pending = emit(0, pending)
            elif low >= _HALF:
                pending = emit(1, pending)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break

            low = (low << 1) & (_FULL_RANGE - 1)
            high = ((high << 1) & (_FULL_RANGE - 1)) | 1

    pending += 1
    if low < _QUARTER:
        pending = emit(0, pending)
    else:
        pending = emit(1, pending)

    return writer.to_bytes()


def _binary_arithmetic_decode(payload: bytes, p0_freqs: np.ndarray, total_freq: int, num_symbols: int) -> np.ndarray:
    if num_symbols == 0:
        return np.array([], dtype=np.uint8)

    low = 0
    high = _FULL_RANGE - 1
    reader = _BitReader(payload)

    code = 0
    for _ in range(32):
        code = ((code << 1) & (_FULL_RANGE - 1)) | reader.read()

    out = np.empty(num_symbols, dtype=np.uint8)

    for i in range(num_symbols):
        p0 = int(p0_freqs[i])
        rng = high - low + 1

        scaled = ((code - low + 1) * total_freq - 1) // rng
        split = low + (rng * p0 // total_freq)

        if scaled < p0:
            out[i] = 0
            high = split - 1
        else:
            out[i] = 1
            low = split

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                code -= _QUARTER
            else:
                break

            low = (low << 1) & (_FULL_RANGE - 1)
            high = ((high << 1) & (_FULL_RANGE - 1)) | 1
            code = ((code << 1) & (_FULL_RANGE - 1)) | reader.read()

    return out


def _context_p0_freqs(symbols: np.ndarray, total_freq: int, context_bits: int) -> np.ndarray:
    if symbols.size == 0:
        return np.array([], dtype=np.uint16)

    num_states = 1 << context_bits
    mask = num_states - 1
    count0 = np.ones(num_states, dtype=np.int64)
    count1 = np.ones(num_states, dtype=np.int64)
    state = 0

    freqs = np.empty(symbols.size, dtype=np.uint16)
    for i, bit in enumerate(symbols.tolist()):
        total = int(count0[state] + count1[state])
        p0 = float(count0[state]) / float(total)
        freqs[i] = np.uint16(np.clip(np.rint(p0 * (total_freq - 2)) + 1, 1, total_freq - 1))

        if bit == 0:
            count0[state] += 1
        else:
            count1[state] += 1
        state = ((state << 1) | int(bit)) & mask

    return freqs


def _binary_arithmetic_decode_context(payload: bytes, total_freq: int, num_symbols: int, context_bits: int) -> np.ndarray:
    if num_symbols == 0:
        return np.array([], dtype=np.uint8)

    low = 0
    high = _FULL_RANGE - 1
    reader = _BitReader(payload)

    code = 0
    for _ in range(32):
        code = ((code << 1) & (_FULL_RANGE - 1)) | reader.read()

    out = np.empty(num_symbols, dtype=np.uint8)

    num_states = 1 << context_bits
    mask = num_states - 1
    count0 = np.ones(num_states, dtype=np.int64)
    count1 = np.ones(num_states, dtype=np.int64)
    state = 0

    for i in range(num_symbols):
        total = int(count0[state] + count1[state])
        p0 = float(count0[state]) / float(total)
        p0_freq = int(np.clip(np.rint(p0 * (total_freq - 2)) + 1, 1, total_freq - 1))

        rng = high - low + 1
        scaled = ((code - low + 1) * total_freq - 1) // rng
        split = low + (rng * p0_freq // total_freq)

        if scaled < p0_freq:
            out[i] = 0
            high = split - 1
        else:
            out[i] = 1
            low = split

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                code -= _QUARTER
            else:
                break

            low = (low << 1) & (_FULL_RANGE - 1)
            high = ((high << 1) & (_FULL_RANGE - 1)) | 1
            code = ((code << 1) & (_FULL_RANGE - 1)) | reader.read()

        bit = int(out[i])
        if bit == 0:
            count0[state] += 1
        else:
            count1[state] += 1
        state = ((state << 1) | bit) & mask

    return out


@dataclass
class BitstreamTensor:
    """Container for packed bitstream plus metadata needed for decoding."""

    payload: bytes
    shape: tuple[int, ...]
    device: str
    context_bits: int = 8
    frequency_precision: int = 12


@dataclass
class EntropyOutput:
    """Output from the learned entropy model."""

    bitstream: BitstreamTensor
    rate_bpp: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor


class EntropyEncoder(nn.Module):
    """Binary arithmetic encoder with adaptive context probabilities."""

    def forward(self, symbols: torch.Tensor, probabilities: torch.Tensor | None = None) -> BitstreamTensor:
        if symbols.dtype != torch.float32 and symbols.dtype != torch.float64:
            symbols = symbols.float()

        binary = ((symbols.detach().cpu().numpy() + 1.0) * 0.5).astype(np.uint8)
        # Side-info-free coding: probabilities are reconstructed from decoded context.
        # probabilities argument is accepted for API compatibility but not used by coder.
        _ = probabilities
        context_bits = 8
        precision = 12
        total = 1 << precision
        p0_freqs = _context_p0_freqs(binary.reshape(-1), total_freq=total, context_bits=context_bits)

        payload = _binary_arithmetic_encode(binary.reshape(-1), p0_freqs.reshape(-1), total)
        return BitstreamTensor(
            payload=payload,
            shape=tuple(binary.shape),
            device=str(symbols.device),
            context_bits=context_bits,
            frequency_precision=precision,
        )


class EntropyDecoder(nn.Module):
    """Binary arithmetic decoder for {-1,+1} symbols."""

    def forward(self, bitstream: BitstreamTensor) -> torch.Tensor:
        total_symbols = int(np.prod(bitstream.shape))
        total = 1 << int(bitstream.frequency_precision)
        decoded = _binary_arithmetic_decode_context(
            bitstream.payload,
            total_freq=total,
            num_symbols=total_symbols,
            context_bits=int(bitstream.context_bits),
        )
        symbols = decoded.reshape(bitstream.shape).astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(symbols)


class LearnedEntropyModel(nn.Module):
    """Predict a Bernoulli prior over binary symbols and estimate rate in bpp.

    The model is trained end-to-end from the latent tensor before binarization
    and drives arithmetic coding probabilities for server-side bitstream output.
    """

    def __init__(self, latent_channels: int = 512, bits: int = 32, hidden_channels: int = 256) -> None:
        super().__init__()
        self.bits = bits
        self.prior = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, bits, kernel_size=1, bias=False),
        )
        self.encoder = EntropyEncoder()

    def forward(
        self,
        latent: torch.Tensor,
        symbols: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> EntropyOutput:
        if latent.ndim != 4 or symbols.ndim != 4:
            raise ValueError("latent and symbols must both be NCHW tensors.")
        if latent.shape[0] != symbols.shape[0] or latent.shape[2:] != symbols.shape[2:]:
            raise ValueError("latent and symbols must have matching batch and spatial dimensions.")

        logits = self.prior(latent)
        probabilities = torch.sigmoid(logits)

        target = ((symbols.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
        per_symbol_nll = F.binary_cross_entropy_with_logits(logits, target, reduction="none") / math.log(2.0)
        pixels = max(1, image_shape[0] * image_shape[1])
        rate_bpp = per_symbol_nll.sum(dim=(1, 2, 3)).mean() / pixels

        bitstream = self.encoder(symbols, probabilities)
        return EntropyOutput(
            bitstream=bitstream,
            rate_bpp=rate_bpp,
            logits=logits,
            probabilities=probabilities,
        )
