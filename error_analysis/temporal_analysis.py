"""Temporal metrics for error analysis.

This module focuses on:
- Frame-to-frame drift
- Error propagation across time
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from compute_metrics import load_rgb_image, mae, mse


def _residual_mse(gt: np.ndarray, pred: np.ndarray) -> float:
	residual = pred - gt
	return float(np.mean(residual * residual))


def compute_temporal_metrics(
	matched: list[tuple[str, Path, Path, int | None]],
) -> list[dict[str, object]]:
	"""Compute per-transition temporal metrics.

	Each row corresponds to transition (t-1 -> t).
	"""
	if len(matched) < 2:
		return []

	rows: list[dict[str, object]] = []

	prev_name, prev_gt_path, prev_dec_path, _ = matched[0]
	prev_gt = load_rgb_image(prev_gt_path)
	prev_dec = load_rgb_image(prev_dec_path)
	prev_residual_mse = _residual_mse(prev_gt, prev_dec)

	for i in range(1, len(matched)):
		name, gt_path, dec_path, frame_idx = matched[i]
		gt = load_rgb_image(gt_path)
		dec = load_rgb_image(dec_path)

		if gt.shape != dec.shape or gt.shape != prev_gt.shape or dec.shape != prev_dec.shape:
			raise ValueError(
				"Shape mismatch across temporal window: "
				f"prev_gt={prev_gt.shape}, prev_dec={prev_dec.shape}, "
				f"gt={gt.shape}, dec={dec.shape}"
			)

		gt_delta = gt - prev_gt
		dec_delta = dec - prev_dec

		drift_mse = mse(gt_delta, dec_delta)
		drift_mae = mae(gt_delta, dec_delta)
		gt_delta_mse = float(np.mean(gt_delta * gt_delta))
		dec_delta_mse = float(np.mean(dec_delta * dec_delta))

		residual_mse_curr = _residual_mse(gt, dec)
		if prev_residual_mse <= 0.0:
			propagation_gain = float("inf") if residual_mse_curr > 0.0 else 1.0
		else:
			propagation_gain = residual_mse_curr / prev_residual_mse

		residual_prev = prev_dec - prev_gt
		residual_curr = dec - gt
		persistence_mse = float(np.mean((residual_curr - residual_prev) ** 2))

		rows.append(
			{
				"transition_index": i,
				"prev_frame_name": prev_name,
				"frame_name": name,
				"frame_index": frame_idx,
				"drift_mse": drift_mse,
				"drift_mae": drift_mae,
				"gt_delta_mse": gt_delta_mse,
				"dec_delta_mse": dec_delta_mse,
				"residual_mse_prev": prev_residual_mse,
				"residual_mse_curr": residual_mse_curr,
				"propagation_gain": propagation_gain,
				"propagation_delta": residual_mse_curr - prev_residual_mse,
				"persistence_mse": persistence_mse,
			}
		)

		prev_name = name
		prev_gt = gt
		prev_dec = dec
		prev_residual_mse = residual_mse_curr

	return rows


def summarize_temporal_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
	"""Aggregate temporal metrics into a summary dictionary."""
	if not rows:
		return {
			"num_transitions": 0,
			"avg_drift_mse": float("nan"),
			"avg_drift_mae": float("nan"),
			"avg_propagation_gain": float("nan"),
			"avg_propagation_delta": float("nan"),
			"avg_persistence_mse": float("nan"),
			"fraction_gain_gt_1": float("nan"),
		}

	drift_mse_vals = np.array([float(r["drift_mse"]) for r in rows], dtype=np.float64)
	drift_mae_vals = np.array([float(r["drift_mae"]) for r in rows], dtype=np.float64)
	prop_gain_vals = np.array([float(r["propagation_gain"]) for r in rows], dtype=np.float64)
	prop_delta_vals = np.array([float(r["propagation_delta"]) for r in rows], dtype=np.float64)
	persist_vals = np.array([float(r["persistence_mse"]) for r in rows], dtype=np.float64)

	finite_gain = prop_gain_vals[np.isfinite(prop_gain_vals)]
	if finite_gain.size > 0:
		fraction_gain_gt_1 = float(np.mean(finite_gain > 1.0))
		avg_gain = float(np.mean(finite_gain))
	else:
		fraction_gain_gt_1 = float("nan")
		avg_gain = float("nan")

	return {
		"num_transitions": len(rows),
		"avg_drift_mse": float(np.mean(drift_mse_vals)),
		"avg_drift_mae": float(np.mean(drift_mae_vals)),
		"avg_propagation_gain": avg_gain,
		"avg_propagation_delta": float(np.mean(prop_delta_vals)),
		"avg_persistence_mse": float(np.mean(persist_vals)),
		"fraction_gain_gt_1": fraction_gain_gt_1,
	}


def _heatmap_to_rgb(heat_norm: np.ndarray) -> np.ndarray:
	"""Convert normalized heatmap [0,1] to RGB pseudo-color."""
	r = np.clip((heat_norm - 0.00) / 0.35, 0.0, 1.0)
	g = np.clip((heat_norm - 0.25) / 0.35, 0.0, 1.0)
	b = np.clip((heat_norm - 0.55) / 0.35, 0.0, 1.0)
	return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def create_temporal_drift_overlays(
	matched: list[tuple[str, Path, Path, int | None]],
	temporal_rows: list[dict[str, object]],
	overlay_dir: Path,
	top_k: int = 20,
	overlay_alpha: float = 0.55,
) -> list[str]:
	"""Create transition-level overlays where temporal drift is concentrated.

	The overlay is drawn on decoded frame t with a pseudo-color heatmap for:
	abs((dec_t - dec_t-1) - (gt_t - gt_t-1)).
	"""
	if len(matched) < 2 or not temporal_rows:
		return []

	overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
	overlay_dir.mkdir(parents=True, exist_ok=True)

	sorted_rows = sorted(temporal_rows, key=lambda r: float(r["drift_mse"]), reverse=True)
	if top_k > 0:
		sorted_rows = sorted_rows[:top_k]
	selected = {int(r["transition_index"]) for r in sorted_rows}

	saved_paths: list[str] = []

	prev_name, prev_gt_path, prev_dec_path, _ = matched[0]
	prev_gt = load_rgb_image(prev_gt_path)
	prev_dec = load_rgb_image(prev_dec_path)

	for i in range(1, len(matched)):
		name, gt_path, dec_path, _ = matched[i]
		if i not in selected:
			prev_name = name
			prev_gt = load_rgb_image(gt_path)
			prev_dec = load_rgb_image(dec_path)
			continue

		gt = load_rgb_image(gt_path)
		dec = load_rgb_image(dec_path)

		if gt.shape != dec.shape or gt.shape != prev_gt.shape or dec.shape != prev_dec.shape:
			raise ValueError(
				"Shape mismatch while creating overlays: "
				f"prev_gt={prev_gt.shape}, prev_dec={prev_dec.shape}, "
				f"gt={gt.shape}, dec={dec.shape}"
			)

		drift_delta = (dec - prev_dec) - (gt - prev_gt)
		drift_map = np.mean(np.abs(drift_delta), axis=2)

		scale = float(np.percentile(drift_map, 99.0))
		if scale <= 0.0:
			heat_norm = np.zeros_like(drift_map, dtype=np.float32)
		else:
			heat_norm = np.clip(drift_map / scale, 0.0, 1.0).astype(np.float32)

		heat_rgb = _heatmap_to_rgb(heat_norm)
		overlay = np.clip((1.0 - overlay_alpha) * dec + overlay_alpha * heat_rgb, 0.0, 255.0).astype(
			np.uint8
		)

		stem = Path(name).stem
		overlay_path = overlay_dir / f"transition_{i:04d}_{stem}_overlay.png"
		Image.fromarray(overlay).save(overlay_path)
		saved_paths.append(str(overlay_path))

		prev_name = name
		prev_gt = gt
		prev_dec = dec

	return saved_paths
