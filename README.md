# Swift: Adaptive Video Streaming with Layered Neural Codecs

Implementation of the **NSDI 2022** paper: "Swift: Adaptive Video Streaming with Layered Neural Codecs". 

This project implements a motion-aware, multi-bitrate architecture with an adaptive ABR scheduler (Section 4.2) designed for real-time streaming on heterogeneous hardware.

---

## 🚀 Key Features

- **Layered Neural Codec**: A 5-level bitrate autoencoder that compresses video residuals.
- **Multi-Exit Decoder**: A client-side decoder with 5 spatial "early exits" to adapt to GPU latency.
- **Section 4.2 ABR Logic**: Real-time adaptation based on a Quality-Computation-Rate (RD-C) matrix.
- **Hardware Telemetry**: Native probes for Battery level, GPU utilization, and Buffer occupancy.
- **Motion-Awareness**: Integrated optical flow caching for high-speed motion-compensated warping.
- **Real Networking**: Socket-based server-client transmission framework using a length-prefixed protocol.

---

## 📂 Project Structure

- `codec/autoencoder/`: Server-side multi-level encoder and motion logic.
- `codec/singleshot/`: Client-side adaptive decoder and ABR policy.
- `streamer/`: Networking implementation (Server/Client nodes) and real-time telemetry.
- `scripts/`: Tools for frame extraction, hardware profiling, and performance evaluation.
- `data/original_frames/`: Storage for raw video frames.
- `flows/`: Cache for calculated motion vectors.
- `models/`: Storage for trained `.pth` checkpoints.

---

## 🛠️ Getting Started

### 1. Prerequisites
```bash
pip install torch torchvision psutil pillow numpy scikit-image opencv-python
```

### 2. Data Preparation
Extract frames from your source video:
```bash
python scripts/extract_frames.py --video path/to/video.mp4 --output-dir data/original_frames
```

### 3. Hardware Profiling (Calibration)
Calibrate the ABR policy to your specific device's GPU/CPU performance:
```bash
python scripts/profile_hardware.py
```
This generates `configs/hardware_profile.json`.

---

## 🏋️ Training Workflow

Swift uses a two-stage training process.

### Stage 1: Multi-Level Autoencoder (Server)
Trains the model to compress motion-compensated residuals.
```bash
python codec/autoencoder/train_script.py --train-dir data/original_frames --epochs 50 --batch-size 4
```

### Stage 2: Singleshot Decoder (Client)
Trains the multi-exit decoder using the frozen autoencoder from Stage 1.
```bash
python -m codec.singleshot.train_singleshot
```

---

## 📺 Running the Streamer

### Simulation Mode (Single Process)
Run the full end-to-end pipeline with simulated network fluctuations:
```bash
python pipeline.py
```

### Real Streaming Mode (Server-Client)
1. **Start the Server** (on your powerful machine):
   ```bash
   python -m streamer.server_demo
   ```
2. **Start the Client** (on your playback device):
   ```bash
   python -m streamer.client_demo
   ```

---

## 📊 Evaluation & Metrics
Generate PSNR, SSIM, and Latency reports across the RD-C spectrum:
```bash
python scripts/evaluate.py
```
Results are saved to `outputs/evaluation/metrics.json`.

---

## 📝 Citation
```bibtex
@inproceedings {278366,
title = {Swift: Adaptive Video Streaming with Layered Neural Codecs},
author={Dasari, Mallesham and Kahatapitiya, Kumara and Das, Samir R. and Balasubramanian, Aruna and Samaras, Dimitris},
booktitle = {19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
year = {2022},
address = {Renton, WA},
url = {https://www.usenix.org/conference/nsdi22/presentation/dasari},
publisher = {USENIX Association},
month = apr,
}
```
