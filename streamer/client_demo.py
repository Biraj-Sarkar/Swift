import torch
import os
import time
from .network_node import SwiftClient
from codec.singleshot.model import SwiftDecoder, SwiftAdaptationPolicy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client starting on {device}...")

    # 1. Load Models
    decoder = SwiftDecoder(v_compress=True).to(device)
    sd_path = "models/singleshot.pth"
    if os.path.exists(sd_path):
        decoder.load_state_dict(torch.load(sd_path, map_location=device))
        print("Loaded decoder weights.")
    decoder.eval()

    # 2. ABR Policy
    policy = SwiftAdaptationPolicy(target_fps=30, buffer_max=10.0)

    # 3. Networking
    client = SwiftClient(host='127.0.0.1', port=5000)
    sock = client.connect()

    # Recurrent States for SwiftDecoder
    h1 = h2 = h3 = h4 = (torch.zeros(1, 512, 16, 16).to(device), torch.zeros(1, 512, 16, 16).to(device))

    print("Starting video playback loop...")
    bandwidth_mbps = 5.0 # Initial guess

    try:
        for i in range(1000): # Stream up to 1000 frames
            # ABR Decision
            telemetry = client.get_hardware_telemetry()
            quality_level, exit_point = policy.decide_config(
                telemetry['gpu_latency'],
                bandwidth_mbps
            )

            # Send request to server
            client.send_data(sock, {'quality_level': quality_level})

            # Receive bitstream
            data = client.receive_data(sock)
            if data is None: break

            bitstream = [chunk.to(device) for chunk in data['bitstream']]

            # Decode
            start_time = time.time()
            with torch.no_grad():
                result = decoder(
                    bitstream_chunks=bitstream,
                    h1=h1, h2=h2, h3=h3, h4=h4,
                    quality_level=len(bitstream),
                    exit_at=exit_point
                )
                h1, h2, h3, h4 = result['states']

            decode_latency = (time.time() - start_time) * 1000

            # Update telemetry for next frame
            client.update_local_metrics(gpu_latency=decode_latency, buffer=policy.buffer_occupancy)

            # Simulated playout
            policy.update_buffer(downloaded_sec=1/30, played_sec=1/30)

            if i % 10 == 0:
                print(f"Frame {data['frame_idx']} | Latency: {decode_latency:.1f}ms | Level: {len(bitstream)} | Exit: {exit_point}")

    finally:
        print("Client disconnected.")
        sock.close()

if __name__ == "__main__":
    main()
