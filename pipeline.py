"""
Swift High-Fidelity Live Video Pipeline.
Simulates a real-world streaming session with Section 4.2 ABR logic.
"""

import os
import time
import warp
import torch
import torch.nn.functional as F
from torchvision import io, transforms
from PIL import Image

from codec.autoencoder.model import MultiLevelAutoencoder
from codec.autoencoder.common import warp
from codec.autoencoder.motion import MotionVectorGenerator
from codec.singleshot.model import SwiftDecoder, SwiftAdaptationPolicy
from streamer.network_node import SwiftClient, SwiftServer

def load_frames_from_dir(directory, limit=30):
    """Loads extracted PNG frames as a list of tensors."""
    frames = []
    if not os.path.exists(directory):
        print(f"[Error] Directory {directory} not found. Using dummy frames.")
        return [torch.randn(3, 256, 256) for _ in range(limit)]

    files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])[:limit]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    for f in files:
        img = Image.open(os.path.join(directory, f)).convert('RGB')
        frames.append(transform(img))
    return frames

def run_live_session(frame_dir="data/original_frames", weights_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swift Live Session (Device: {device}) ---")

    # 1. INITIALIZE MODELS (Assumed pre-trained)
    autoencoder = MultiLevelAutoencoder(num_levels=5).to(device)
    decoder = SwiftDecoder(v_compress=True).to(device)

    ae_path = os.path.join(weights_dir, "autoencoder.pth")
    sd_path = os.path.join(weights_dir, "singleshot.pth")

    if os.path.exists(ae_path): autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    if os.path.exists(sd_path): decoder.load_state_dict(torch.load(sd_path, map_location=device))

    autoencoder.eval()
    decoder.eval()

    # 2. INITIALIZE CLIENT BRAIN (Section 4.2)
    mv_gen = MotionVectorGenerator()
    policy = SwiftAdaptationPolicy(target_fps=30, buffer_max=10.0)

    # 3. INITIALIZE NETWORKING & TELEMETRY
    client = SwiftClient()
    # Initial state
    client.update_local_metrics(gpu_latency=20.0, buffer=0.0)

    bandwidth_mbps = 5.0  # Simulated starting bandwidth

    # Load Real Frames
    frames = load_frames_from_dir(frame_dir)
    num_frames = len(frames)

    # Correctly initialize recurrent states with matching spatial scales
    h1, h2, h3, h4 = decoder.init_states(1, device)

    print(f"Processing {num_frames} frames...")

    for i in range(1, num_frames - 1):
        # --- CLIENT: TELEMETRY & DECISION ---
        telemetry = client.get_hardware_telemetry()

        # Update policy based on "Device" telemetry
        policy.buffer_occupancy = telemetry['buffer_seconds']

        # Simulated Buffer consumption/production logic for policy
        played_sec = 1.0 / 30.0
        downloaded_sec = 1.0 if i % 30 == 0 else 0.0
        policy.update_buffer(downloaded_sec, played_sec)

        # ABR Decision using Telemetry
        quality_level, exit_point = policy.decide_config(
            telemetry['gpu_latency'],
            bandwidth_mbps
        )

        # --- SERVER: MOTION & ENCODING ---
        past, curr, future = frames[i-1], frames[i], frames[i+1]
        mv_past, mv_future = mv_gen.generate_triplet_mvs(past, curr, future)

        with torch.no_grad():
            _, outputs, _ = autoencoder(
                curr.unsqueeze(0).to(device),
                ref_frames=(past.unsqueeze(0).to(device), future.unsqueeze(0).to(device)),
                motion_vectors=(mv_past, mv_future)
            )
            bitstream = [out.symbols for out in outputs[:quality_level]]

        # --- CLIENT: DECODING ---
        start_time = time.time()
        with torch.no_grad():
            # Extract UNet features for temporal fusion
            combined_context = torch.cat([warp(past.unsqueeze(0).to(device), mv_past),
                                         warp(future.unsqueeze(0).to(device), mv_future)], dim=1)
            unet_features = autoencoder.unet(combined_context)
            prediction = (warp(past.unsqueeze(0).to(device), mv_past) +
                         warp(future.unsqueeze(0).to(device), mv_future)) / 2.0

            result = decoder(
                bitstream_chunks=bitstream,
                h1=h1, h2=h2, h3=h3, h4=h4,
                quality_level=len(bitstream),
                exit_at=exit_point,
                prediction=prediction,
                context_unet=unet_features
            )
            h1, h2, h3, h4 = result['states']

        # UPDATE TELEMETRY for next iteration
        current_latency = (time.time() - start_time) * 1000
        client.update_local_metrics(
            gpu_latency=current_latency,
            buffer=policy.buffer_occupancy
        )

        # Simulation: Randomly fluctuate bandwidth to test ABR
        if i == 10: bandwidth_mbps = 1.0 # Network drop
        if i == 20: bandwidth_mbps = 8.0 # Network recovery

        if i % 5 == 0:
            print(f"Frame {i:03d} | Net: {bandwidth_mbps}Mbps | "
                  f"GPU: {telemetry['gpu_latency']:.1f}ms (Load: {telemetry['gpu_load']}%) | "
                  f"Bat: {telemetry['battery_level']}% | "
                  f"Buf: {policy.buffer_occupancy:.1f}s | Level: {quality_level} | Exit: {exit_point}")

    print("--- Session Complete ---")

if __name__ == "__main__":
    run_live_session()
