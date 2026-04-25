"""Motion Vector Generation module for Swift.
Uses Farneback Optical Flow as a robust baseline for motion estimation.
"""

import torch
import numpy as np
import cv2
import os
import hashlib

class MotionVectorGenerator:
    """Generates optical flow (Motion Vectors) between frames."""

    def __init__(self, use_cuda=True, cache_dir=None):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        if cache_dir is None:
            # More robust project root detection: look for 'pipeline.py' or 'codec' folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            while current_dir != os.path.dirname(current_dir): # stop at root
                if "pipeline.py" in os.listdir(current_dir) or "codec" in os.listdir(current_dir):
                    break
                current_dir = os.path.dirname(current_dir)
            self.cache_dir = os.path.join(current_dir, "flows")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, frame_a, frame_b):
        """Generates a unique filename based on frame content hash."""
        # Use a sample of pixels from the frame for hashing (center 64x64)
        h, w = frame_a.shape[-2:]
        ch, cw = h // 2, w // 2
        sample_a = frame_a[:, ch-32:ch+32, cw-32:cw+32].cpu().numpy().tobytes()
        sample_b = frame_b[:, ch-32:ch+32, cw-32:cw+32].cpu().numpy().tobytes()

        h_str = hashlib.md5(sample_a + sample_b).hexdigest()
        return os.path.join(self.cache_dir, f"flow_{h_str}.pth")

    def clear_cache(self):
        """Wipes all cached flows."""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Flow cache at {self.cache_dir} cleared.")

    def compute_flow(self, prev_frame, curr_frame, use_cache=True):
        """
        Computes flow from prev_frame to curr_frame.
        Input: [C, H, W] Tensors (0-1 float)
        Output: [2, H, W] Tensor
        """
        cache_path = self._get_cache_path(prev_frame, curr_frame)
        if use_cache and os.path.exists(cache_path):
            return torch.load(cache_path, map_location=self.device)

        # Convert to grayscale numpy for OpenCV
        prev_np = (prev_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        curr_np = (curr_frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)

        # Farneback algorithm for dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Convert back to torch [2, H, W]
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float().to(self.device)

        if use_cache:
            torch.save(flow_tensor, cache_path)

        return flow_tensor

    def generate_triplet_mvs(self, past, current, future, use_cache=True):
        """
        Generates MVs for a frame triplet: (past -> current) and (future -> current).
        """
        mv_past = self.compute_flow(past, current, use_cache=use_cache)
        mv_future = self.compute_flow(future, current, use_cache=use_cache)
        return mv_past.unsqueeze(0), mv_future.unsqueeze(0)
