import os
import torch
import json
import time
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec.autoencoder.model import MultiLevelAutoencoder
from codec.autoencoder.motion import MotionVectorGenerator
from codec.singleshot.model import SwiftDecoder
from evaluation_analysis import (
    run_nsdi_suite,
    evaluate_against_baselines,
    plot_evaluation_results,
    run_color_analysis,
    run_pixel_analysis,
    run_temporal_analysis
)

def load_frame(path, transform):
    img = Image.open(path).convert('RGB')
    return transform(img)

@torch.no_grad()
def run_evaluation(frames_dir="data/original_frames", models_dir="models", output_dir="outputs/evaluation"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swift Performance Evaluation (Device: {device}) ---")

    # 1. Initialize Models
    autoencoder = MultiLevelAutoencoder(num_levels=5).to(device)
    decoder = SwiftDecoder(v_compress=True).to(device)

    ae_path = os.path.join(models_dir, "autoencoder.pth")
    sd_path = os.path.join(models_dir, "singleshot.pth")

    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print("Loaded Autoencoder weights.")
    if os.path.exists(sd_path):
        decoder.load_state_dict(torch.load(sd_path, map_location=device))
        print("Loaded Singleshot Decoder weights.")

    autoencoder.eval()
    decoder.eval()

    mv_gen = MotionVectorGenerator()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 2. Get Frame Sequence
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if len(frame_files) < 3:
        print("Error: Not enough frames for evaluation.")
        return

    # 3. RD-C Profile configurations
    configs = [(5, 'final'), (3, 'ee_1_4'), (1, 'ee_1_16')]
    all_results = []

    recon_frames_dir = Path(output_dir) / "reconstructed_frames"
    recon_frames_dir.mkdir(parents=True, exist_ok=True)

    for q_level, exit_at in configs:
        print(f"Evaluating Config: Level={q_level}, Exit={exit_at}...")
        metrics = {'psnr': [], 'ssim': [], 'latency_ms': [], 'bpp': []}

        # Correctly initialize recurrent states with matching spatial scales
        h1, h2, h3, h4 = decoder.init_states(1, device)

        for i in range(1, len(frame_files) - 1):
            past = load_frame(os.path.join(frames_dir, frame_files[i-1]), transform)
            curr = load_frame(os.path.join(frames_dir, frame_files[i]), transform)
            future = load_frame(os.path.join(frames_dir, frame_files[i+1]), transform)

            mv_past, mv_future = mv_gen.generate_triplet_mvs(past, curr, future)

            # Use Autoencoder to get bitstream and context features
            with torch.no_grad():
                past_t, curr_t, future_t = past.unsqueeze(0).to(device), curr.unsqueeze(0).to(device), future.unsqueeze(0).to(device)
                from codec.autoencoder.common import warp
                warped_past = warp(past_t, mv_past.to(device))
                warped_future = warp(future_t, mv_future.to(device))
                combined_context = torch.cat([warped_past, warped_future], dim=1)
                unet_features = autoencoder.unet(combined_context)
                prediction = (warped_past + warped_future) / 2.0

                _, outputs, rate_bpp = autoencoder(curr_t, ref_frames=(past_t, future_t), motion_vectors=(mv_past.to(device), mv_future.to(device)))
                bitstream = [out.symbols for out in outputs[:q_level]]

            start_time = time.time()
            result = decoder(bitstream, h1, h2, h3, h4, q_level, exit_at, prediction=prediction, context_unet=unet_features)
            latency = (time.time() - start_time) * 1000
            h1, h2, h3, h4 = result['states']

            orig_np = (curr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            recon_img_tensor = result['output'].squeeze(0).cpu().clamp(0, 1)
            recon_img = transforms.ToPILImage()(recon_img_tensor).resize((256, 256))
            recon_np = np.array(recon_img)

            # Save reconstructed frame for later deep analysis
            if exit_at == 'final':
                recon_img.save(recon_frames_dir / f"recon_{frame_files[i]}")

            metrics['psnr'].append(psnr_metric(orig_np, recon_np))
            metrics['ssim'].append(ssim_metric(orig_np, recon_np, channel_axis=2))
            metrics['latency_ms'].append(latency)
            metrics['bpp'].append(float(rate_bpp))

        summary = {
            'config': {'quality_level': q_level, 'exit_at': exit_at},
            'avg_psnr': round(np.mean(metrics['psnr']), 2),
            'avg_ssim': round(np.mean(metrics['ssim']), 4),
            'avg_latency': round(np.mean(metrics['latency_ms']), 2),
            'avg_bpp': round(np.mean(metrics['bpp']), 4)
        }
        all_results.append(summary)

    # 4. Save Swift Results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(all_results, f, indent=4)

    # 5. Deep Error Analysis (Color, Pixel, Temporal)
    print("\n--- Running Deep Error Analysis ---")
    sample_idx = len(frame_files) // 2
    target_sample = Path(frames_dir) / frame_files[sample_idx]
    pred_sample = recon_frames_dir / f"recon_{frame_files[sample_idx]}"

    if pred_sample.exists():
        pixel_results = run_pixel_analysis(target_sample, pred_sample, Path(output_dir) / "heatmaps")
        color_results = run_color_analysis(target_sample, pred_sample, Path(output_dir))

        target_paths = [Path(frames_dir) / f for f in frame_files[1:-1]]
        pred_paths = [recon_frames_dir / f"recon_{f}" for f in frame_files[1:-1]]
        temporal_results = run_temporal_analysis(target_paths, pred_paths)

        deep_analysis = {
            "pixel": pixel_results,
            "color": color_results,
            "temporal": temporal_results
        }
        with open(os.path.join(output_dir, "deep_analysis.json"), 'w') as f:
            json.dump(deep_analysis, f, indent=4)
        print("Deep analysis complete.")

    # 6. Final Suite
    print("\n--- Executing Final Suite ---")
    run_nsdi_suite(decoder, None, device)
    evaluate_against_baselines(frames_dir)
    plot_evaluation_results()

if __name__ == "__main__":
    run_evaluation()
