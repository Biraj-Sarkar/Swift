"""Evaluation based on NSDI'22 Swift paper:
- Quality vs. Bitrate (Rate-Distortion)
- Adaptation Latency
- Decoding Complexity (FPS)
"""

import torch
import time
import numpy as np
from pathlib import Path
from .compute_metrics import load_rgb_image

def measure_decoding_speed(model, input_codes, quality_level, num_runs=100):
    """Measures decoding speed in FPS as per Swift paper analysis."""
    model.eval()
    device = next(model.parameters()).device
    batch = input_codes[0].shape[0] if isinstance(input_codes, (list, tuple)) else input_codes.shape[0]

    def run_once():
        states = model.init_states(batch, device)
        return model(input_codes, *states, quality_level=quality_level)

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = run_once()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = run_once()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    return fps

def compute_rd_curve(quality_metrics: list[dict], bitrates: list[float]):
    """Computes Rate-Distortion statistics."""
    # quality_metrics: list of dicts containing PSNR, SSIM etc. for each quality level
    # bitrates: list of bitrates in bpp or kbps for each level

    curve = []
    for q, b in zip(quality_metrics, bitrates):
        curve.append({
            "bitrate": b,
            "psnr": q["psnr"],
            "ms_ssim": q.get("ms_ssim", 0)
        })
    return curve

def run_nsdi_suite(model, test_data_loader, device):
    """Executes the standard evaluation suite from the paper."""
    results = {
        "levels": {},
        "adaptation_latency": "To be measured with streamer",
    }

    for q in range(1, 6):
        fps = 30.0 # Placeholder for actual measurement logic
        # Here you would loop through data, compute metrics and aggregate
        results["levels"][q] = {
            "fps": fps,
            "avg_psnr": 0.0,
            "avg_bpp": 0.0
        }

    return results
