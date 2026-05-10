"""
Actual Baseline Evaluation for Swift Codec.
Calculates H.264 PSNR using FFmpeg and provides hooks for SVC/SHVC.
"""

import os
import torch
import json
import re
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

PSNR_IDENTICAL_FRAME_DB = 100.0

def _finite_psnr(target, pred):
    value = psnr_metric(target, pred)
    if np.isfinite(value):
        return float(value)
    return PSNR_IDENTICAL_FRAME_DB

def _first_frame_number(frames_dir):
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    match = re.search(r"frame_(\d+)\.png$", frame_files[0].name)
    if not match:
        raise ValueError(f"Unexpected frame filename: {frame_files[0].name}")
    return int(match.group(1))

def get_h264_baseline_metrics(frames_dir, crf=23, framerate=24):
    """
    Actually runs FFmpeg to compress the frames and calculates real PSNR.
    """
    frames_dir = Path(frames_dir)
    output_mp4 = "temp_baseline_h264.mp4"
    recon_dir = Path("data/baseline_recon_h264")
    os.makedirs(recon_dir, exist_ok=True)

    try:
        start_number = _first_frame_number(frames_dir)

        # 1. Compress frames to MP4 (H.264)
        # Extracted frames are named frame_000001.png on the ffmpeg path, so
        # explicitly tell ffmpeg where the image sequence starts.
        cmd_compress = [
            'ffmpeg', '-y',
            '-framerate', str(framerate),
            '-start_number', str(start_number),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', output_mp4
        ]
        subprocess.run(cmd_compress, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoded_size_bytes = os.path.getsize(output_mp4)

        # 2. Decompress back to frames to measure distortion
        for old_frame in recon_dir.glob("frame_*.png"):
            old_frame.unlink()
        cmd_decompress = [
            'ffmpeg', '-y', '-i', output_mp4, str(recon_dir / 'frame_%04d.png')
        ]
        subprocess.run(cmd_decompress, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Calculate PSNR comparing original vs reconstructed
        orig_files = sorted(list(frames_dir.glob("*.png")))
        recon_files = sorted(list(recon_dir.glob("*.png")))

        psnrs = []
        for o_file, r_file in zip(orig_files, recon_files):
            o_img = np.array(Image.open(o_file).convert('RGB'))
            r_img = np.array(Image.open(r_file).convert('RGB'))
            # Standardize size for comparison
            if o_img.shape != r_img.shape:
                r_img = np.array(Image.fromarray(r_img).resize((o_img.shape[1], o_img.shape[0])))
            psnrs.append(_finite_psnr(o_img, r_img))

        # Cleanup
        if os.path.exists(output_mp4): os.remove(output_mp4)

        if not psnrs:
            return None

        height, width = np.array(Image.open(orig_files[0]).convert('RGB')).shape[:2]
        bpp = (encoded_size_bytes * 8.0) / max(1, len(psnrs) * height * width)

        return {
            "psnr": round(np.mean(psnrs), 2),
            "bpp": round(float(bpp), 4)
        }
    except Exception as e:
        print(f"H.264 Baseline Error: {e}. Ensure ffmpeg is installed.")
        return None

def evaluate_against_baselines(frames_dir="data/original_frames", swift_metrics_path="outputs/evaluation/metrics.json"):
    print("--- Running Actual Baseline Analysis ---")

    # 1. H.264 (Actually Measured)
    h264_metrics = get_h264_baseline_metrics(frames_dir)

    # 2. SVC / SHVC (Dynamic Lookup)
    # Check if user has run the baseline binaries and saved results
    baselines = {}
    if h264_metrics:
        baselines["H.264 (Measured)"] = h264_metrics

    # Check for SVC (JSVM) output
    svc_result = Path("baselines/svc/results.json")
    if svc_result.exists():
        with open(svc_result, 'r') as f:
            baselines["SVC (JSVM)"] = json.load(f)
    else:
        # If no binary results, we add the target from the paper as a reference line
        baselines["SVC (Paper Target)"] = {"psnr": 33.1, "bpp": 0.18}

    if os.path.exists(swift_metrics_path):
        with open(swift_metrics_path, 'r') as f:
            swift_data = json.load(f)
    else:
        swift_data = []

    report = {
        "swift_results": swift_data,
        "baselines": baselines,
        "environment": {
            "frames_dir": str(frames_dir),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
    }

    output_path = "outputs/evaluation/baseline_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4, allow_nan=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluate_against_baselines()
