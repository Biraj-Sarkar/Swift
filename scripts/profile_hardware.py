import torch
import time
import json
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec.singleshot.model import SwiftDecoder

def profile_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Swift Hardware Profiling (Device: {device}) ---")

    decoder = SwiftDecoder(v_compress=True).to(device)
    decoder.eval()

    # Input dimensions (standardized to 256x256 in dataset)
    # Bitstream chunks are usually tensors of shape (1, bits, 16, 16)
    # Let's assume bits=32 as per train_script.py default
    bits = 32

    quality_levels = [1, 2, 3, 4, 5]
    exit_points = ['ee_1_16', 'ee_1_8', 'ee_1_4', 'ee_1_2', 'final']

    results = {}

    # Warm up
    print("Warming up...")
    dummy_chunks = [torch.randn(1, bits, 16, 16).to(device) for _ in range(5)]
    h1, h2, h3, h4 = decoder.init_states(1, device)

    for _ in range(10):
        with torch.no_grad():
            _ = decoder(dummy_chunks, h1, h2, h3, h4, quality_level=5, exit_at='final')

    print("Profiling iterations...")
    for q in quality_levels:
        results[q] = {}
        chunks = [torch.randn(1, bits, 16, 16).to(device) for _ in range(q)]
        for e in exit_points:
            # Measure time over multiple runs for stability
            iterations = 50
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

            latencies = []
            for _ in range(iterations):
                h1, h2, h3, h4 = decoder.init_states(1, device)

                if device.type == 'cuda':
                    start_event.record()
                    with torch.no_grad():
                        _ = decoder(chunks, h1, h2, h3, h4, quality_level=q, exit_at=e)
                    end_event.record()
                    torch.cuda.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                else:
                    start_time = time.time()
                    with torch.no_grad():
                        _ = decoder(chunks, h1, h2, h3, h4, quality_level=q, exit_at=e)
                    latencies.append((time.time() - start_time) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            results[q][e] = round(avg_latency, 2)
            print(f"Quality {q} | Exit {e:6} | Latency: {avg_latency:.2f} ms")

    # Save to a JSON file for the AdaptationPolicy to use
    output_path = "configs/hardware_profile.json"
    os.makedirs("configs", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nProfiling complete. Results saved to {output_path}")
    print("This data can now be used to calibrate the SwiftAdaptationPolicy (Section 4.2).")

if __name__ == "__main__":
    profile_decoder()
