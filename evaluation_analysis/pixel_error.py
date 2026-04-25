import numpy as np
import cv2
from pathlib import Path
from .compute_metrics import load_rgb_image

def compute_pixel_error_map(target: np.ndarray, pred: np.ndarray):
    """Compute the absolute difference map."""
    return np.abs(target - pred)

def save_error_heatmap(error_map, output_path: Path):
    """Save a heatmap of the error."""
    # Sum across channels or take max
    error_gray = np.mean(error_map, axis=2).astype(np.uint8)
    heatmap = cv2.applyColorMap(error_gray, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path), heatmap)

def run_pixel_analysis(target_path: Path, pred_path: Path, output_dir: Path):
    target = load_rgb_image(target_path)
    pred = load_rgb_image(pred_path)

    error_map = compute_pixel_error_map(target, pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_error_heatmap(error_map, output_dir / f"pixel_error_{target_path.stem}.png")

    # Simple statistics
    mean_error = np.mean(error_map)
    max_error = np.max(error_map)
    return {"mean_pixel_error": float(mean_error), "max_pixel_error": float(max_error)}
