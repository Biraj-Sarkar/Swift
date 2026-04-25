import numpy as np
from pathlib import Path
from .compute_metrics import load_rgb_image

def run_temporal_analysis(frame_paths: list[Path], pred_paths: list[Path]):
    """Analyze error stability across a sequence of frames."""
    psnrs = []

    for f_path, p_path in zip(frame_paths, pred_paths):
        target = load_rgb_image(f_path)
        pred = load_rgb_image(p_path)

        mse = np.mean((target - pred) ** 2)
        if mse > 0:
            psnr = 10 * np.log10((255.0**2) / mse)
        else:
            psnr = 100.0
        psnrs.append(psnr)

    psnrs = np.array(psnrs)
    results = {
        "psnr_variance": float(np.var(psnrs)),
        "psnr_min": float(np.min(psnrs)),
        "psnr_max": float(np.max(psnrs)),
        "psnr_mean": float(np.mean(psnrs))
    }

    return results
