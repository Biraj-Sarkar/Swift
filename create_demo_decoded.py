"""Create synthetic decoded frames for demonstration."""

from pathlib import Path

import cv2
import numpy as np


def create_demo_decoded_frames(
	gt_dir: Path,
	output_dir: Path,
	quality: int = 85,
) -> None:
	"""Create synthetic decoded frames by JPEG compression simulation.

	Args:
		gt_dir: Directory with GT frames.
		output_dir: Directory where decoded frames will be saved.
		quality: JPEG quality (0-100, lower=more artifacts).
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	gt_frames = sorted(gt_dir.glob("*.png"))
	print(f"Creating {len(gt_frames)} synthetic decoded frames (quality={quality})...")

	for i, gt_path in enumerate(gt_frames):
		frame = cv2.imread(str(gt_path))
		if frame is None:
			print(f"  Skipped {gt_path.name} (read error)")
			continue

		_, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
		decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

		out_path = output_dir / gt_path.name
		cv2.imwrite(str(out_path), decoded)

		if (i + 1) % 20 == 0:
			print(f"  Processed {i + 1}/{len(gt_frames)} frames...")

	print(f"Complete: {len(gt_frames)} decoded frames saved to {output_dir}")


if __name__ == "__main__":
	gt_dir = Path("data/original_frames")
	output_dir = Path("outputs/decoded_frames")

	create_demo_decoded_frames(gt_dir, output_dir, quality=85)
