"""Extract frames from a video file using OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
	video_path: Path,
	output_dir: Path,
	frame_skip: int = 1,
	max_frames: int | None = None,
) -> int:
	"""Extract frames from video and save as PNG.

	Args:
		video_path: Path to input video file.
		output_dir: Directory where frames will be saved.
		frame_skip: Extract every Nth frame (1=all, 2=every other, etc).
		max_frames: Maximum frames to extract (None=all).

	Returns:
		Number of frames extracted.
	"""
	if not video_path.exists():
		raise FileNotFoundError(f"Video not found: {video_path}")

	output_dir.mkdir(parents=True, exist_ok=True)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video: {video_path}")

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	print(f"Video: {video_path.name}")
	print(f"  Total frames: {total_frames}")
	print(f"  FPS: {fps:.2f}")
	print(f"  Resolution: {width}x{height}")
	print(f"  Frame skip: {frame_skip}")
	if max_frames:
		print(f"  Max frames to extract: {max_frames}")

	frame_idx = 0
	extracted = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		if frame_idx % frame_skip == 0:
			if max_frames and extracted >= max_frames:
				break

			out_path = output_dir / f"frame_{extracted:06d}.png"
			cv2.imwrite(str(out_path), frame)
			extracted += 1

			if extracted % 50 == 0:
				print(f"  Extracted {extracted} frames...")

		frame_idx += 1

	cap.release()
	print(f"Extraction complete: {extracted} frames saved to {output_dir}")
	return extracted


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract frames from video file.")
	parser.add_argument("--video", required=True, help="Path to input video file")
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Directory where frames will be saved",
	)
	parser.add_argument(
		"--frame-skip",
		type=int,
		default=1,
		help="Extract every Nth frame (default: 1=all frames)",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Maximum frames to extract (default: all)",
	)

	args = parser.parse_args()
	video_path = Path(args.video)
	output_dir = Path(args.output_dir)

	extract_frames(
		video_path=video_path,
		output_dir=output_dir,
		frame_skip=args.frame_skip,
		max_frames=args.max_frames,
	)


if __name__ == "__main__":
	main()
