import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# --- Core Modules ---

class ResNetBlock(nn.Module):
    """Refined residual block for high-quality reconstruction."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.conv(x)

class EarlyExit(nn.Module):
    """Spatial exit head to output RGB at intermediate resolutions."""
    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.upscale = nn.PixelShuffle(upscale_factor)
        # Channels after pixel shuffle = in_channels / (upscale_factor^2)
        out_channels = in_channels // (upscale_factor ** 2)
        self.conv = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, x):
        x = self.upscale(x)
        return torch.tanh(self.conv(x)) / 2

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, hidden_kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.input_to_state = nn.Conv2d(in_channels, 4 * out_channels, kernel_size, stride, padding)
        self.state_to_state = nn.Conv2d(out_channels, 4 * out_channels, hidden_kernel_size, 1, hidden_kernel_size // 2)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = self.input_to_state(x) + self.state_to_state(h_prev)
        i, f, o, g = torch.split(combined, self.out_channels, dim=1)
        c_next = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

# --- Adaptation Policy ---

class SwiftAdaptationPolicy:
    """
    Implements Section 4.2: Adapting ABR for Layered Neural Codecs.
    Uses a Quality-Computation Matrix, GPU Capacity, and Buffer Occupancy.
    """
    def __init__(self, target_fps=30, buffer_max=20.0, segment_duration=1.0):
        self.target_fps = target_fps
        self.buffer_max = buffer_max # Seconds
        self.segment_duration = segment_duration # Seconds
        self.buffer_occupancy = 0.0 # Current seconds in buffer

        # 1. Quality-Computation Matrix Q(l, e)
        self.q_matrix = {
            1: {'ee_1_16': 25, 'ee_1_8': 26, 'ee_1_4': 27, 'ee_1_2': 28, 'final': 29},
            2: {'ee_1_16': 26, 'ee_1_8': 28, 'ee_1_4': 29, 'ee_1_2': 31, 'final': 32},
            3: {'ee_1_16': 26, 'ee_1_8': 29, 'ee_1_4': 31, 'ee_1_2': 33, 'final': 35},
            4: {'ee_1_16': 26, 'ee_1_8': 29, 'ee_1_4': 32, 'ee_1_2': 35, 'final': 38},
            5: {'ee_1_16': 26, 'ee_1_8': 29, 'ee_1_4': 33, 'ee_1_2': 37, 'final': 42},
        }

        # 2. Computation Cost Matrix C(q, e)
        # Attempt to load actual hardware profile
        self.c_matrix = self._load_hardware_profile()

    def _load_hardware_profile(self):
        """Loads results from scripts/profile_hardware.py if available."""
        # Standardize path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        profile_path = os.path.join(project_root, "configs", "hardware_profile.json")

        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    # Convert string keys from JSON back to ints for levels
                    return {int(k): v for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Failed to load hardware profile: {e}")

        # Fallback default (Level 5 baseline)
        print("Using default computation cost matrix.")
        return {
            5: {
                'ee_1_16': 5.0,
                'ee_1_8': 8.0,
                'ee_1_4': 15.0,
                'ee_1_2': 22.0,
                'final': 32.0
            }
        }

    def update_buffer(self, downloaded_seconds, played_seconds):
        """Track segments buffered but not yet played."""
        self.buffer_occupancy = max(0, self.buffer_occupancy + downloaded_seconds - played_seconds)
        self.buffer_occupancy = min(self.buffer_occupancy, self.buffer_max)

    def decide_config(self, gpu_latency_ms, bandwidth_mbps):
        """
        Solves Section 4.2 Objective: Maximize Q while respecting Bandwidth and GPU Capacity.
        """
        best_q = -1
        best_config = (1, 'ee_1_16')

        buffer_factor = min(1.0, self.buffer_occupancy / 5.0)
        max_compute_ms = (1000.0 / self.target_fps)

        for l in range(1, 6):
            # Bandwidth check
            if (l * 1.2) > bandwidth_mbps * (0.5 + 0.5 * buffer_factor):
                continue

            # Get cost for this level, fallback to level 5 if missing
            level_costs = self.c_matrix.get(l, self.c_matrix.get(5))

            for e, predicted_cost in level_costs.items():
                # We use measured gpu_latency_ms as a real-time scaler to handle thermal throttling
                # We normalize against the level 5 'final' cost from profiling
                baseline_final = level_costs.get('final', 32.0)
                real_time_scaler = gpu_latency_ms / baseline_final

                adjusted_cost = predicted_cost * real_time_scaler

                if adjusted_cost < max_compute_ms:
                    q_val = self.q_matrix[l].get(e, 0)
                    if q_val > best_q:
                        best_q = q_val
                        best_config = (l, e)

        return best_config

# --- The Swift Singleshot Decoder ---

class SwiftDecoder(nn.Module):
    """
    Multi-headed, multi-exit decoder for adaptive streaming.
    Supports 5 Bitrate levels and 5 Computational exits.
    """
    def __init__(self, bits_per_head=32, v_compress=True):
        super().__init__()
        self.v_compress = v_compress

        # 1. Bitstream Adaptation Heads (Rate Adaptation)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bits_per_head, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, 1),
                ResNetBlock(512)
            ) for _ in range(5)
        ])

        # 2. Main Recurrent Backbone
        self.rnn1 = ConvLSTMCell(512, 512, 3, 1, 1, 1)
        self.rnn2 = ConvLSTMCell(128, 512, 3, 1, 1, 1) # Upsampled
        self.rnn3 = ConvLSTMCell(128, 256, 3, 1, 1, 3) # Upsampled
        self.rnn4 = ConvLSTMCell(64, 128, 3, 1, 1, 3)  # Upsampled

        # 3. Spatial Early Exits (Device/Hardware Adaptation)
        self.ee1 = EarlyExit(512, upscale_factor=16) # 1/16 scale
        self.ee2 = EarlyExit(512, upscale_factor=8)  # 1/8 scale
        self.ee3 = EarlyExit(256, upscale_factor=4)  # 1/4 scale
        self.ee4 = EarlyExit(128, upscale_factor=2)  # 1/2 scale

        self.conv_final = nn.Conv2d(32, 3, 1) # Full Scale

    @staticmethod
    def init_states(batch_size, device, height=256, width=256):
        """Create recurrent states for the fixed spatial hierarchy."""
        return (
            (torch.zeros(batch_size, 512, height // 16, width // 16, device=device),
             torch.zeros(batch_size, 512, height // 16, width // 16, device=device)),
            (torch.zeros(batch_size, 512, height // 8, width // 8, device=device),
             torch.zeros(batch_size, 512, height // 8, width // 8, device=device)),
            (torch.zeros(batch_size, 256, height // 4, width // 4, device=device),
             torch.zeros(batch_size, 256, height // 4, width // 4, device=device)),
            (torch.zeros(batch_size, 128, height // 2, width // 2, device=device),
             torch.zeros(batch_size, 128, height // 2, width // 2, device=device)),
        )

    def forward(self, bitstream_chunks, h1, h2, h3, h4,
                quality_level=5, exit_at='final', prediction=None, context_unet=None, return_all_exits=False):
        """
        Args:
            bitstream_chunks: List of 5 tensors [N, bits, H, W]
            quality_level: 1-5 (How many bitstream heads to aggregate)
            exit_at: 'ee_1_16', 'ee_1_8', 'ee_1_4', 'ee_1_2', 'final'
            prediction: The motion-warped prediction frame [N, 3, H, W]
            context_unet: Features from neighboring frames for interpolation
            return_all_exits: If True, returns all intermediate reconstructions (for training)
        """
        results = {'states': (h1, h2, h3, h4), 'all_exits': {}}

        if prediction is None:
            # Fallback to gray if no prediction provided (not recommended)
            prediction = torch.zeros((h1[0].shape[0], 3, h1[0].shape[2]*16, h1[0].shape[3]*16), device=h1[0].device) + 0.5

        # 1. Aggregate bits (Rate adaptation)
        x = 0
        for i in range(quality_level):
            x = x + self.heads[i](bitstream_chunks[i])

        # Level 1: Latent Space (1/16)
        h1 = self.rnn1(x, h1)
        delta_1_16 = self.ee1(h1[0])
        pred_1_16 = F.interpolate(prediction, size=delta_1_16.shape[-2:], mode='bilinear', align_corners=False)
        out_1_16 = (pred_1_16 + delta_1_16).clamp(0, 1)

        results['all_exits']['ee_1_16'] = out_1_16
        if exit_at == 'ee_1_16' and not return_all_exits:
            results['states'] = (h1, h2, h3, h4)
            results['output'] = out_1_16
            results['scale'] = 1/16
            return results

        # Level 2: Upsample x2 (1/8)
        x = F.pixel_shuffle(h1[0], 2)
        if self.v_compress and context_unet:
            x = x + context_unet[0]
        h2 = self.rnn2(x, h2)
        delta_1_8 = self.ee2(h2[0])
        pred_1_8 = F.interpolate(prediction, size=delta_1_8.shape[-2:], mode='bilinear', align_corners=False)
        out_1_8 = (pred_1_8 + delta_1_8).clamp(0, 1)

        results['all_exits']['ee_1_8'] = out_1_8
        if exit_at == 'ee_1_8' and not return_all_exits:
            results['states'] = (h1, h2, h3, h4)
            results['output'] = out_1_8
            results['scale'] = 1/8
            return results

        # Level 3: Upsample x4 (1/4)
        x = F.pixel_shuffle(h2[0], 2)
        if self.v_compress and context_unet:
            x = x + context_unet[1]
        h3 = self.rnn3(x, h3)
        delta_1_4 = self.ee3(h3[0])
        pred_1_4 = F.interpolate(prediction, size=delta_1_4.shape[-2:], mode='bilinear', align_corners=False)
        out_1_4 = (pred_1_4 + delta_1_4).clamp(0, 1)

        results['all_exits']['ee_1_4'] = out_1_4
        if exit_at == 'ee_1_4' and not return_all_exits:
            results['states'] = (h1, h2, h3, h4)
            results['output'] = out_1_4
            results['scale'] = 1/4
            return results

        # Level 4: Upsample x8 (1/2)
        x = F.pixel_shuffle(h3[0], 2)
        if self.v_compress and context_unet:
            x = x + context_unet[2]
        h4 = self.rnn4(x, h4)
        delta_1_2 = self.ee4(h4[0])
        pred_1_2 = F.interpolate(prediction, size=delta_1_2.shape[-2:], mode='bilinear', align_corners=False)
        out_1_2 = (pred_1_2 + delta_1_2).clamp(0, 1)

        results['all_exits']['ee_1_2'] = out_1_2
        if exit_at == 'ee_1_2' and not return_all_exits:
            results['states'] = (h1, h2, h3, h4)
            results['output'] = out_1_2
            results['scale'] = 1/2
            return results

        # Level 5: Final x16 Reconstruction (Full)
        x = F.pixel_shuffle(h4[0], 2)
        delta_final = torch.tanh(self.conv_final(x)) / 2
        out_final = (prediction + delta_final).clamp(0, 1)

        results['all_exits']['final'] = out_final
        results['output'] = out_final
        results['scale'] = 1.0
        results['states'] = (h1, h2, h3, h4)

        return results

    def save_model(self, directory: str = None, filename: str = "singleshot.pth") -> str:
        """Saves the model state dict to the project's models directory."""
        if directory is None:
            # Calculate project root (2 levels up from codec/singleshot/model.py)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            directory = os.path.join(project_root, "models")

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save(self.state_dict(), path)
        print(f"Singleshot decoder model saved to {path}")
        return path
