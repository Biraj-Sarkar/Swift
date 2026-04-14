"""Training utilities for the Swift-style multi-level autoencoder.

This module uses L1 reconstruction loss, matching the original Swift training
objective for residual reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .model import MultiLevelAutoencoder


@dataclass
class TrainStepResult:
    loss: float
    reconstruction_loss: float
    rate_loss: float


def compute_l1_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute Swift-style L1 reconstruction loss."""
    return F.l1_loss(reconstruction, target)


def _extract_images(batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Empty batch received.")
        batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        raise TypeError("Batch must be a tensor or tuple/list whose first element is a tensor.")
    return batch


def train_step(
    model: MultiLevelAutoencoder,
    batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    context: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    grad_clip: float | None = None,
    rate_weight: float = 1e-2,
) -> TrainStepResult:
    """Run one optimization step with L1 reconstruction and learned rate loss."""
    model.train()

    x = _extract_images(batch).to(device=device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    reconstruction, _, rate_loss = model(x, context=context, use_entropy_decode=False)
    reconstruction_loss = compute_l1_reconstruction_loss(reconstruction, x)
    loss = reconstruction_loss + rate_weight * rate_loss

    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return TrainStepResult(
        loss=float(loss.detach().item()),
        reconstruction_loss=float(reconstruction_loss.detach().item()),
        rate_loss=float(rate_loss.detach().item()),
    )


def fit_one_epoch(
    model: MultiLevelAutoencoder,
    dataloader: Iterable[torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    context: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    grad_clip: float | None = None,
    rate_weight: float = 1e-2,
) -> float:
    """Train for one epoch and return mean L1 loss."""
    total_loss = 0.0
    num_steps = 0

    for batch in dataloader:
        step = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            context=context,
            grad_clip=grad_clip,
            rate_weight=rate_weight,
        )
        total_loss += step.loss
        num_steps += 1

    if num_steps == 0:
        raise ValueError("Dataloader is empty.")

    return total_loss / num_steps
