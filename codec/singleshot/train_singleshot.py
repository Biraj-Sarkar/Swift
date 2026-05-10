import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import SwiftDecoder
from ..autoencoder.common import warp
from ..autoencoder.model import MultiLevelAutoencoder
from ..dataset import SwiftDataset

def train_singleshot(frames_dir="data/original_frames", ae_path="models/autoencoder.pth", epochs=50, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swift Singleshot Training (Device: {device}) ---")

    # 1. Load Pre-trained Autoencoder (Frozen)
    print(f"Loading Autoencoder from: {ae_path}")

    if not os.path.exists(ae_path):
        raise FileNotFoundError(
            f"Pre-trained autoencoder not found at: {ae_path}"
        )

    autoencoder = MultiLevelAutoencoder(num_levels=5).to(device)
    autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    print("Pre-trained Autoencoder loaded successfully")
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    # 2. Initialize Decoder and Optimizer
    print("Initializing Singleshot Decoder...")
    decoder = SwiftDecoder(v_compress=True).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    # 3. Load Real Dataset
    print(f"Loading dataset from: {frames_dir}")

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}"
        )

    dataset = SwiftDataset(frames_dir=frames_dir, calc_mvs=True)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Dataset size: {len(dataset)} frame triplets")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    # 4. Training Loop
    print(f"Starting training on {len(dataset)} frame triplets...")

    exit_weights = {'ee_1_16': 0.1, 'ee_1_8': 0.2, 'ee_1_4': 0.3, 'ee_1_2': 0.4, 'final': 1.0}
    quality_weights = {1: 0.35, 2: 0.45, 3: 0.65, 4: 0.85, 5: 1.0}
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0

        for batch_idx, (past, curr, future, mv_past, mv_future) in enumerate(dataloader):
            # Move to device
            past, curr, future = past.to(device, non_blocking=True), curr.to(device, non_blocking=True), future.to(device, non_blocking=True)
            mv_past, mv_future = mv_past.to(device, non_blocking=True), mv_future.to(device, non_blocking=True)

            optimizer.zero_grad()

            # A. Get Bitstream from frozen Autoencoder (using motion)
            with torch.no_grad():
                _, outputs, _ = autoencoder(
                    curr,
                    ref_frames=(past, future),
                    motion_vectors=(mv_past, mv_future)
                )
                bitstream = [out.symbols for out in outputs]

            # B. Forward pass through Decoder
            # Extract UNet features from frozen Autoencoder for context-aware reconstruction
            with torch.no_grad():
                warped_past = warp(past, mv_past)
                warped_future = warp(future, mv_future)
                combined_context = torch.cat([warped_past, warped_future], dim=1)
                unet_features = autoencoder.unet(combined_context)
                prediction = (warped_past + warped_future) / 2.0

            # C. Multi-quality, multi-exit reconstruction loss. The decoder is
            # evaluated at levels 1/3/5, so train those partial bitstream paths.
            loss = 0
            for quality_level, quality_weight in quality_weights.items():
                q_h1, q_h2, q_h3, q_h4 = decoder.init_states(
                    curr.shape[0],
                    device,
                    height=curr.shape[-2],
                    width=curr.shape[-1],
                )
                results = decoder(
                    bitstream[:quality_level],
                    q_h1,
                    q_h2,
                    q_h3,
                    q_h4,
                    quality_level=quality_level,
                    prediction=prediction,
                    context_unet=unet_features,
                    return_all_exits=True,
                )

                for exit_name, reconstructed in results['all_exits'].items():
                    target = F.interpolate(curr, size=reconstructed.shape[-2:], mode='bilinear', align_corners=False)
                    loss += quality_weight * exit_weights[exit_name] * F.l1_loss(reconstructed, target)

            loss = loss / sum(quality_weights.values())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        checkpoint_path = "models/singleshot_latest.pth"

        torch.save(
            decoder.state_dict(),
            checkpoint_path
        )

    print("Singleshot training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/original_frames"
    )

    parser.add_argument(
        "--ae-path",
        type=str,
        default="models/autoencoder.pth"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    args = parser.parse_args()

    train_singleshot(
        frames_dir=args.frames_dir,
        ae_path=args.ae_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
