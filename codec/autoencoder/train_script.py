"""Runnable training script for the Swift-style multi-level autoencoder.

Usage examples:

python codec/autoencoder/train_script.py --train-dir data/original_frames --epochs 20
python -m codec.autoencoder.train_script --train-dir data/original_frames --val-split 0.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

try:
    from .model import MultiLevelAutoencoder
    from .train import train_step
    from ..dataset import SwiftDataset
except ImportError:
    from model import MultiLevelAutoencoder
    from train import train_step
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import SwiftDataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str | Path, transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Train directory not found: {self.root}")

        self.paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.root} with extensions: {sorted(IMAGE_EXTS)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swift-style autoencoder with L1 loss.")
    parser.add_argument("--train-dir", required=True, type=str, help="Directory with training images.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize/crop square size; use multiples of 16.")
    parser.add_argument("--levels", type=int, default=5, help="Number of coding levels.")
    parser.add_argument("--bits", type=int, default=32, help="Bits per level for binarizer symbols.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split in [0,1).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--rate-weight", type=float, default=1e-2, help="Weight for learned entropy rate loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-dir", type=str, default="codec/autoencoder/checkpoints", help="Checkpoint directory.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = SwiftDataset(args.train_dir, transform=transform, calc_mvs=True)

    val_loader: DataLoader | None = None
    if args.val_split > 0.0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        if val_size == 0 or train_size == 0:
            raise ValueError("val-split creates empty train or validation partition.")
        train_set, val_set = random_split(
            dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_metrics(
    model: MultiLevelAutoencoder,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_l1 = 0.0
    running_rate = 0.0
    steps = 0

    for batch in val_loader:
        past, curr, future, mv_past, mv_future = [b.to(device, non_blocking=True) for b in batch]
        reconstruction, _, rate_bpp = model(
            curr,
            ref_frames=(past, future),
            motion_vectors=(mv_past, mv_future)
        )
        loss = F.l1_loss(reconstruction, curr)
        running_l1 += float(loss.item())
        running_rate += float(rate_bpp.item())
        steps += 1

    return running_l1 / max(1, steps), running_rate / max(1, steps)


def maybe_load_checkpoint(model: MultiLevelAutoencoder, optimizer: torch.optim.Optimizer, resume_path: str) -> tuple[int, float]:
    if not resume_path:
        return 0, float("inf")

    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    print(f"Resumed from {resume_path} at epoch {start_epoch}.")
    return start_epoch, best_val


def save_checkpoint(
    model: MultiLevelAutoencoder,
    optimizer: torch.optim.Optimizer,
    save_dir: Path,
    epoch: int,
    train_l1: float,
    val_l1: float | None,
    best_val: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_l1": train_l1,
        "val_l1": val_l1,
        "best_val": best_val,
    }
    torch.save(payload, save_dir / "latest.pt")

    if val_l1 is not None and val_l1 <= best_val:
        torch.save(payload, save_dir / "best.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16 for this architecture.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(args)

    model = MultiLevelAutoencoder(num_levels=args.levels, bits=args.bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    start_epoch, best_val = maybe_load_checkpoint(model, optimizer, args.resume)

    print(
        f"Starting training on {device}; train batches: {len(train_loader)}; "
        f"val batches: {0 if val_loader is None else len(val_loader)}"
    )

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        running_recon = 0.0
        running_rate = 0.0
        steps = 0

        for batch in train_loader:
            step = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                rate_weight=args.rate_weight,
            )
            running_loss += step.loss
            running_recon += step.reconstruction_loss
            running_rate += step.rate_loss
            steps += 1

        train_loss = running_loss / max(1, steps)
        train_l1 = running_recon / max(1, steps)
        train_rate = running_rate / max(1, steps)

        val_l1 = None
        val_rate = None
        if val_loader is not None:
            val_l1, val_rate = evaluate_metrics(model, val_loader, device)
            best_val = min(best_val, val_l1)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            save_dir=save_dir,
            epoch=epoch,
            train_l1=train_loss,
            val_l1=val_l1,
            best_val=best_val,
        )

        # Also save the model
        model.save_model()

        if val_l1 is None:
            print(
                f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | "
                f"train_l1={train_l1:.6f} | train_rate={train_rate:.6f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | train_l1={train_l1:.6f} | "
                f"train_rate={train_rate:.6f} | val_l1={val_l1:.6f} | val_rate={val_rate:.6f}"
            )


if __name__ == "__main__":
    main()