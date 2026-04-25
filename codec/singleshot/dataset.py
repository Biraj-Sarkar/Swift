import os
import glob
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2

def get_loader(is_train, root, mv_dir, args):
    dset = ImageFolder(is_train=is_train, root=root, args=args)
    loader = data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
        num_workers=2
    )
    return loader

def default_loader(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    if h % 16 != 0 or w % 16 != 0:
        img = img[:(h//16)*16, :(w//16)*16]
    return img

def np_to_torch(img):
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()

class ImageFolder(data.Dataset):
    def __init__(self, is_train, root, args):
        self.is_train = is_train
        self.root = root
        self.args = args
        self.patch = args.patch
        self.num_crops = args.num_crops
        self.imgs = sorted(glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.jpg")))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = default_loader(path)

        if self.is_train:
            crops = []
            h, w, _ = img.shape
            for _ in range(self.num_crops):
                y = random.randint(0, h - self.patch)
                x = random.randint(0, w - self.patch)
                crop = img[y:y+self.patch, x:x+self.patch].copy()
                crop = crop.astype(np.float32) / 255.0
                crops.append(np_to_torch(crop))
            return crops, torch.zeros(1), path
        else:
            img = img.astype(np.float32) / 255.0
            return np_to_torch(img), torch.zeros(1), path

    def __len__(self):
        return len(self.imgs)
