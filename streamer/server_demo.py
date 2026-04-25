import torch
import os
import time
from .network_node import SwiftServer
from codec.autoencoder.model import MultiLevelAutoencoder
from codec.autoencoder.motion import MotionVectorGenerator
from torchvision import transforms
from PIL import Image

def load_frame(frame_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(Image.open(frame_path).convert('RGB'))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Server starting on {device}...")

    # 1. Load Models
    autoencoder = MultiLevelAutoencoder(num_levels=5).to(device)
    ae_path = "models/autoencoder.pth"
    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print("Loaded autoencoder weights.")
    autoencoder.eval()

    mv_gen = MotionVectorGenerator()

    # 2. Networking
    server = SwiftServer(host='0.0.0.0', port=5000)
    conn = server.start()

    # 3. Stream Data
    frame_dir = "data/original_frames"
    if not os.path.exists(frame_dir):
        print(f"Error: {frame_dir} not found. Please extract frames first.")
        return

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    # We need triplets for motion compensation
    for i in range(1, len(frame_files) - 1):
        # Wait for client request
        request = server.receive_data(conn)
        if request is None: break

        quality_level = request.get('quality_level', 5)
        print(f"Client requested Quality Level: {quality_level}")

        # Load frames
        past = load_frame(os.path.join(frame_dir, frame_files[i-1]))
        curr = load_frame(os.path.join(frame_dir, frame_files[i]))
        future = load_frame(os.path.join(frame_dir, frame_files[i+1]))

        # Generate Motion Vectors
        mv_past, mv_future = mv_gen.generate_triplet_mvs(past, curr, future)

        # Encode
        with torch.no_grad():
            _, outputs, _ = autoencoder(
                curr.unsqueeze(0).to(device),
                ref_frames=(past.unsqueeze(0).to(device), future.unsqueeze(0).to(device)),
                motion_vectors=(mv_past, mv_future)
            )

            # Prepare bitstream up to requested quality
            bitstream = [out.symbols.cpu() for out in outputs[:quality_level]]

        # Send to client
        server.send_data(conn, {
            'frame_idx': i,
            'bitstream': bitstream,
            'quality_level': quality_level
        })

    print("Streaming finished.")
    conn.close()

if __name__ == "__main__":
    main()
