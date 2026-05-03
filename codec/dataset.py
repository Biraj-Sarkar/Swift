import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .autoencoder.motion import MotionVectorGenerator

class SwiftDataset(Dataset):
    """
    Unified Dataset for Swift.
    Returns triplets of frames (past, current, future) and their motion vectors.
    """
    def __init__(self, frames_dir, transform=None, calc_mvs=True):
        self.frames_dir = frames_dir
        self.files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        self.calc_mvs = calc_mvs

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        if calc_mvs:
            # Keep dataset outputs on CPU; training/evaluation loops move batches to device.
            self.mv_gen = MotionVectorGenerator(use_cuda=False)

    def __len__(self):
        # We need at least 3 frames for a triplet
        return max(0, len(self.files) - 2)

    def __getitem__(self, idx):
        # Triplet indices: i-1, i, i+1
        f_past_path = os.path.join(self.frames_dir, self.files[idx])
        f_curr_path = os.path.join(self.frames_dir, self.files[idx+1])
        f_future_path = os.path.join(self.frames_dir, self.files[idx+2])

        # Load and transform images
        past = self.transform(Image.open(f_past_path).convert('RGB'))
        curr = self.transform(Image.open(f_curr_path).convert('RGB'))
        future = self.transform(Image.open(f_future_path).convert('RGB'))

        if self.calc_mvs:
            # generate_triplet_mvs handles the flows/ caching automatically
            mv_past, mv_future = self.mv_gen.generate_triplet_mvs(past, curr, future)
            return past, curr, future, mv_past.squeeze(0), mv_future.squeeze(0)

        return past, curr, future
