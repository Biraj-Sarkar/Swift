"""Core metric utilities for error analysis v1.

This module intentionally keeps only baseline metrics:
- MSE
- MAE
- PSNR
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(image_path) -> np.ndarray:
	"""Load an image as float32 RGB array in [0, 255]."""
	with Image.open(image_path) as img:
		rgb = img.convert("RGB")
		arr = np.asarray(rgb, dtype=np.float32)
	return arr


def mse(target, pred) -> float:
	"""Mean squared error over all pixels/channels."""
	diff = target - pred
	return float(np.mean(diff * diff))


def mae(target, pred) -> float:
	"""Mean absolute error over all pixels/channels."""
	return float(np.mean(np.abs(target - pred)))


def psnr_from_mse(mse_value, peak_value = 255.0) -> float:
	"""Compute PSNR from MSE.

	Returns +inf when MSE is 0 (identical images).
	"""
	if mse_value <= 0.0:
		return float("inf")
	return float(10.0 * np.log10((peak_value * peak_value) / mse_value))


def compute_frame_metrics(target, pred) -> dict[str, float]:
	"""Compute baseline metrics for one frame pair."""
	if target.shape != pred.shape:
		raise ValueError(f"Shape mismatch: target={target.shape}, pred={pred.shape}")

	mse_value = mse(target, pred)
	return {
		"mse": mse_value,
		"mae": mae(target, pred),
		"psnr": psnr_from_mse(mse_value),
	}
