"""Run baseline error analysis between ground-truth and decoded frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_metrics import compute_frame_metrics, load_rgb_image
from temporal_analysis import (
	compute_temporal_metrics,
	create_temporal_drift_overlays,
	summarize_temporal_metrics,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Error analysis v1: computes MSE/MAE/PSNR for matched frame pairs.",
	)
	parser.add_argument("--gt-dir", required=True, help="Ground-truth frames directory")
	parser.add_argument("--decoded-dir", required=True, help="Decoded frames directory")
	parser.add_argument(
		"--output-dir",
		default="error_analysis_outputs",
		help="Directory for CSV/JSON/plot outputs",
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=None,
		help="Optional start frame index (inclusive)",
	)
	parser.add_argument(
		"--end-index",
		type=int,
		default=None,
		help="Optional end frame index (inclusive)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional max number of matched frames to process",
	)
	parser.add_argument(
		"--overlay-top-k",
		type=int,
		default=20,
		help="Number of highest-drift transitions to export as overlay images (<=0 means all)",
	)
	parser.add_argument(
		"--overlay-alpha",
		type=float,
		default=0.55,
		help="Overlay blend alpha in [0, 1]",
	)
	return parser.parse_args()


def _extract_trailing_int(stem: str) -> int | None:
	digits = []
	for ch in reversed(stem):
		if ch.isdigit():
			digits.append(ch)
		else:
			break
	if not digits:
		return None
	return int("".join(reversed(digits)))


def _list_images_by_name(directory: Path) -> dict[str, Path]:
	files: dict[str, Path] = {}
	for path in directory.iterdir():
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			files[path.name] = path
	return files


def _match_frames(
	gt_dir: Path,
	decoded_dir: Path,
	start_index: int | None,
	end_index: int | None,
	limit: int | None,
) -> list[tuple[str, Path, Path, int | None]]:
	gt_files = _list_images_by_name(gt_dir)
	dec_files = _list_images_by_name(decoded_dir)
	common_names = sorted(set(gt_files.keys()) & set(dec_files.keys()))

	matched: list[tuple[str, Path, Path, int | None]] = []
	for name in common_names:
		idx = _extract_trailing_int(Path(name).stem)
		if start_index is not None and idx is not None and idx < start_index:
			continue
		if end_index is not None and idx is not None and idx > end_index:
			continue
		matched.append((name, gt_files[name], dec_files[name], idx))

	if limit is not None:
		matched = matched[:limit]
	return matched


def _safe_mean(values: list[float]) -> float:
	if not values:
		return float("nan")
	return float(np.mean(values))


def _write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["frame_name", "frame_index", "mse", "mae", "psnr"],
		)
		writer.writeheader()
		writer.writerows(rows)


def _write_summary_json(summary_path: Path, summary: dict[str, object]) -> None:
	with summary_path.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)


def _write_temporal_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"transition_index",
				"prev_frame_name",
				"frame_name",
				"frame_index",
				"drift_mse",
				"drift_mae",
				"gt_delta_mse",
				"dec_delta_mse",
				"residual_mse_prev",
				"residual_mse_curr",
				"propagation_gain",
				"propagation_delta",
				"persistence_mse",
			],
		)
		writer.writeheader()
		writer.writerows(rows)


def _plot_psnr(plot_path: Path, rows: list[dict[str, object]]) -> None:
	x = list(range(len(rows)))
	labels = [str(r["frame_name"]) for r in rows]
	y = [float(r["psnr"]) for r in rows]

	plt.figure(figsize=(10, 4.5))
	plt.plot(x, y, linewidth=1.5)
	plt.title("PSNR per Frame")
	plt.xlabel("Frame")
	plt.ylabel("PSNR (dB)")
	if len(rows) <= 30:
		plt.xticks(x, labels, rotation=60, ha="right")
	else:
		plt.xticks([])
	plt.tight_layout()
	plt.savefig(plot_path, dpi=160)
	plt.close()


def _plot_temporal_metrics(plot_path: Path, rows: list[dict[str, object]]) -> None:
	if not rows:
		plt.figure(figsize=(9, 4))
		plt.title("Temporal Metrics")
		plt.text(0.5, 0.5, "No temporal transitions available", ha="center", va="center")
		plt.axis("off")
		plt.tight_layout()
		plt.savefig(plot_path, dpi=160)
		plt.close()
		return

	x = [int(r["transition_index"]) for r in rows]
	drift = [float(r["drift_mse"]) for r in rows]
	prop_gain = [float(r["propagation_gain"]) for r in rows]

	plt.figure(figsize=(10, 5))
	plt.plot(x, drift, label="Drift MSE", linewidth=1.5)
	plt.plot(x, prop_gain, label="Propagation Gain", linewidth=1.2)
	plt.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.7)
	plt.title("Temporal Drift and Error Propagation")
	plt.xlabel("Transition Index (t-1 -> t)")
	plt.ylabel("Metric Value")
	plt.legend()
	plt.tight_layout()
	plt.savefig(plot_path, dpi=160)
	plt.close()


def main() -> None:
	args = _parse_args()

	gt_dir = Path(args.gt_dir)
	decoded_dir = Path(args.decoded_dir)
	output_dir = Path(args.output_dir)

	if not gt_dir.exists() or not gt_dir.is_dir():
		raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
	if not decoded_dir.exists() or not decoded_dir.is_dir():
		raise FileNotFoundError(f"Decoded directory not found: {decoded_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)

	matched = _match_frames(
		gt_dir=gt_dir,
		decoded_dir=decoded_dir,
		start_index=args.start_index,
		end_index=args.end_index,
		limit=args.limit,
	)

	if not matched:
		raise RuntimeError("No matching frames found by filename between the two folders.")

	rows: list[dict[str, object]] = []
	for frame_name, gt_path, dec_path, frame_idx in matched:
		gt = load_rgb_image(gt_path)
		pred = load_rgb_image(dec_path)
		metrics = compute_frame_metrics(gt, pred)

		rows.append(
			{
				"frame_name": frame_name,
				"frame_index": frame_idx,
				"mse": metrics["mse"],
				"mae": metrics["mae"],
				"psnr": metrics["psnr"],
			}
		)

	mse_vals = [float(r["mse"]) for r in rows]
	mae_vals = [float(r["mae"]) for r in rows]
	psnr_vals = [float(r["psnr"]) for r in rows if np.isfinite(float(r["psnr"]))]

	summary: dict[str, object] = {
		"num_frames": len(rows),
		"avg_mse": _safe_mean(mse_vals),
		"avg_mae": _safe_mean(mae_vals),
		"avg_psnr": _safe_mean(psnr_vals),
		"gt_dir": str(gt_dir),
		"decoded_dir": str(decoded_dir),
	}

	temporal_rows = compute_temporal_metrics(matched)
	temporal_summary = summarize_temporal_metrics(temporal_rows)
	summary["temporal"] = temporal_summary

	csv_path = output_dir / "frame_metrics.csv"
	summary_path = output_dir / "summary.json"
	plot_path = output_dir / "psnr_plot.png"
	temporal_csv_path = output_dir / "temporal_metrics.csv"
	temporal_summary_path = output_dir / "temporal_summary.json"
	temporal_plot_path = output_dir / "temporal_plot.png"
	overlay_dir = output_dir / "temporal_overlays"

	_write_csv(csv_path, rows)
	_write_summary_json(summary_path, summary)
	_plot_psnr(plot_path, rows)
	_write_temporal_csv(temporal_csv_path, temporal_rows)
	_write_summary_json(temporal_summary_path, temporal_summary)
	_plot_temporal_metrics(temporal_plot_path, temporal_rows)
	overlay_paths = create_temporal_drift_overlays(
		matched=matched,
		temporal_rows=temporal_rows,
		overlay_dir=overlay_dir,
		top_k=args.overlay_top_k,
		overlay_alpha=args.overlay_alpha,
	)

	print("Analysis complete")
	print(f"Frames processed: {summary['num_frames']}")
	print(f"Transitions analyzed: {temporal_summary['num_transitions']}")
	print(f"Temporal overlays saved: {len(overlay_paths)}")
	print(f"CSV: {csv_path}")
	print(f"Summary: {summary_path}")
	print(f"Plot: {plot_path}")
	print(f"Temporal CSV: {temporal_csv_path}")
	print(f"Temporal summary: {temporal_summary_path}")
	print(f"Temporal plot: {temporal_plot_path}")
	print(f"Temporal overlays dir: {overlay_dir}")


if __name__ == "__main__":
	main()
