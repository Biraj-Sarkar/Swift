import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import SwiftDecoder
from ..autoencoder.common import warp
from ..autoencoder.model import MultiLevelAutoencoder
from ..dataset import SwiftDataset

def train_singleshot(frames_dir="data/original_frames", epochs=50, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swift Singleshot Training (Device: {device}) ---")

    # 1. Load Pre-trained Autoencoder (Frozen)
    autoencoder = MultiLevelAutoencoder(num_levels=5).to(device)
    ae_path = os.path.join("models", "autoencoder.pth")
    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print(f"Loaded pre-trained Autoencoder from {ae_path}")
    else:
        print("Warning: No pre-trained autoencoder found. Training with random AE weights.")
    autoencoder.eval()

    # 2. Initialize Decoder and Optimizer
    decoder = SwiftDecoder(v_compress=True).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    # 3. Load Real Dataset
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} not found. Please extract frames first.")
        return

    dataset = SwiftDataset(frames_dir=frames_dir, calc_mvs=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Training Loop
    print(f"Starting training on {len(dataset)} frame triplets...")

    weights = {'ee_1_16': 0.1, 'ee_1_8': 0.2, 'ee_1_4': 0.3, 'ee_1_2': 0.4, 'final': 1.0}

    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0

        for past, curr, future, mv_past, mv_future in dataloader:
            # Move to device
            past, curr, future = past.to(device), curr.to(device), future.to(device)
            mv_past, mv_future = mv_past.to(device), mv_future.to(device)

            optimizer.zero_grad()

            # A. Get Bitstream from frozen Autoencoder (using motion)
            with torch.no_grad():
                _, outputs, _ = autoencoder(
                    curr,
                    ref_frames=(past, future),
                    motion_vectors=(mv_past, mv_future)
                )
                bitstream = [out.symbols for out in outputs]

            # B. Init states with correct spatial scales
            h1, h2, h3, h4 = decoder.init_states(
                curr.shape[0],
                device,
                height=curr.shape[-2],
                width=curr.shape[-1],
            )

            # C. Forward pass through Decoder
            # Extract UNet features from frozen Autoencoder for context-aware reconstruction
            with torch.no_grad():
                warped_past = warp(past, mv_past)
                warped_future = warp(future, mv_future)
                combined_context = torch.cat([warped_past, warped_future], dim=1)
                unet_features = autoencoder.unet(combined_context)
                prediction = (warped_past + warped_future) / 2.0

            results = decoder(
                bitstream,
                h1,
                h2,
                h3,
                h4,
                prediction=prediction,
                context_unet=unet_features,
                return_all_exits=True,
            )

            # D. Multi-Exit Reconstruction Loss
            loss = 0
            for exit_name, reconstructed in results['all_exits'].items():
                target = F.interpolate(curr, size=reconstructed.shape[-2:], mode='bilinear', align_corners=False)
                loss += weights[exit_name] * F.l1_loss(reconstructed, target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            decoder.save_model()

    print("Singleshot training complete.")

if __name__ == "__main__":
    train_singleshot()
