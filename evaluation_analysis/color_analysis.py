import numpy as np
import cv2
from pathlib import Path
from .compute_metrics import load_rgb_image

def run_color_analysis(target_path: Path, pred_path: Path, output_dir: Path):
    target = load_rgb_image(target_path)
    pred = load_rgb_image(pred_path)

    # Analyze distribution in YUV space
    target_yuv = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2YUV)
    pred_yuv = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2YUV)

    y_error = np.abs(target_yuv[:,:,0].astype(float) - pred_yuv[:,:,0].astype(float))
    u_error = np.abs(target_yuv[:,:,1].astype(float) - pred_yuv[:,:,1].astype(float))
    v_error = np.abs(target_yuv[:,:,2].astype(float) - pred_yuv[:,:,2].astype(float))

    results = {
        "mean_y_error": float(np.mean(y_error)),
        "mean_u_error": float(np.mean(u_error)),
        "mean_v_error": float(np.mean(v_error))
    }

    return results
