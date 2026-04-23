"""Extract frames from a video file using GStreamer or ffmpeg."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import cv2


def _opencv_supports_gstreamer() -> bool:
	build_info = cv2.getBuildInformation()
	for line in build_info.splitlines():
		if line.strip().startswith("GStreamer:"):
			return "YES" in line
	return False


def _make_gstreamer_pipeline(video_path: Path) -> str:
	# Force raw BGR frames into appsink so OpenCV receives a standard array.
	return (
		f'filesrc location={shlex.quote(str(video_path))} ! '
		"parsebin ! vaapidecodebin ! videoconvert ! video/x-raw,format=BGR ! appsink "
		"drop=1 sync=false"
	)


def _parse_ffmpeg_fraction(value: str | None) -> float:
	if not value:
		return 0.0
	try:
		numerator, denominator = value.split("/", maxsplit=1)
		return float(numerator) / float(denominator)
	except (ValueError, ZeroDivisionError):
		try:
			return float(value)
		except ValueError:
			return 0.0


def _print_ffmpeg_video_info(video_path: Path, frame_skip: int, max_frames: int | None) -> None:
	print(f"Video: {video_path.name}")

	ffprobe = shutil.which("ffprobe")
	if ffprobe is None:
		print("  Total frames: unknown")
		print("  FPS: unknown")
		print("  Resolution: unknown")
		print(f"  Frame skip: {frame_skip}")
		if max_frames:
			print(f"  Max frames to extract: {max_frames}")
		return

	command = [
		ffprobe,
		"-v",
		"error",
		"-select_streams",
		"v:0",
		"-show_entries",
		"stream=width,height,avg_frame_rate,nb_frames,duration",
		"-of",
		"json",
		str(video_path),
	]
	try:
		result = subprocess.run(command, check=True, capture_output=True, text=True)
		probe = json.loads(result.stdout)
	except (subprocess.CalledProcessError, json.JSONDecodeError):
		probe = {}

	streams = probe.get("streams", [])
	stream = streams[0] if streams else {}
	fps = _parse_ffmpeg_fraction(stream.get("avg_frame_rate"))
	total_frames = stream.get("nb_frames")
	if total_frames is None and fps > 0:
		try:
			total_frames = str(round(float(stream.get("duration", 0.0)) * fps))
		except ValueError:
			total_frames = "unknown"

	width = stream.get("width")
	height = stream.get("height")
	resolution = f"{width}x{height}" if width and height else "unknown"

	print(f"  Total frames: {total_frames or 'unknown'}")
	if fps > 0:
		print(f"  FPS: {fps:.2f}")
	else:
		print("  FPS: unknown")
	print(f"  Resolution: {resolution}")
	print(f"  Frame skip: {frame_skip}")
	if max_frames:
		print(f"  Max frames to extract: {max_frames}")


def _extract_frames_gstreamer(
	video_path: Path,
	output_dir: Path,
	frame_skip: int,
	max_frames: int | None,
) -> int:
	if not _opencv_supports_gstreamer():
		raise RuntimeError(
			"OpenCV in this venv was built without GStreamer support, so a GStreamer pipeline cannot be used. "
			"Rebuild OpenCV with GStreamer enabled, or use the ffmpeg fallback."
		)
	pipeline = _make_gstreamer_pipeline(video_path)
	cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
	if not cap.isOpened():
		return 0

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
	return extracted


def _extract_frames_ffmpeg_cli(
	video_path: Path,
	output_dir: Path,
	frame_skip: int,
	max_frames: int | None,
) -> int:
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is None:
		raise RuntimeError(
			"OpenCV could not decode the input and the ffmpeg CLI is not installed in PATH. "
			"Install ffmpeg or re-encode the source video to H.264/H.265."
		)

	_print_ffmpeg_video_info(video_path, frame_skip, max_frames)

	existing = {path.name for path in output_dir.glob("frame_*.png")}
	command = [
		ffmpeg,
		"-hide_banner",
		"-loglevel",
		"error",
		"-nostats",
		"-progress",
		"pipe:1",
		"-i",
		str(video_path),
	]

	filters: list[str] = []
	if frame_skip > 1:
		filters.append(f"select='not(mod(n\\,{frame_skip}))'")
	filters.append("format=bgr24")
	if filters:
		command += ["-vf", ",".join(filters)]
	command += ["-vsync", "0"]
	if max_frames is not None:
		command += ["-frames:v", str(max_frames)]
	command += [str(output_dir / "frame_%06d.png")]

	try:
		process = subprocess.Popen(
			command,
			stdout=subprocess.PIPE,
			stderr=subprocess.DEVNULL,
			text=True,
		)
		latest_reported = 0
		assert process.stdout is not None
		for line in process.stdout:
			line = line.strip()
			if not line.startswith("frame="):
				continue
			try:
				current_reported = int(line.split("=", maxsplit=1)[1])
			except ValueError:
				continue
			while current_reported >= latest_reported + 50:
				latest_reported += 50
				print(f"  Extracted {latest_reported} frames...")
			latest_reported = max(latest_reported, current_reported)

		return_code = process.wait()
		if return_code != 0:
			raise subprocess.CalledProcessError(return_code, command)
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(f"ffmpeg CLI failed while decoding {video_path}: {exc}") from exc

	current = {path.name for path in output_dir.glob("frame_*.png")}
	return max(0, len(current - existing))


def _extract_frames_gstreamer_first(
	video_path: Path,
	output_dir: Path,
	frame_skip: int,
	max_frames: int | None,
) -> int:
	extracted = _extract_frames_gstreamer(video_path, output_dir, frame_skip, max_frames)
	if extracted > 0:
		return extracted
	print("GStreamer decode failed; falling back to ffmpeg CLI...")
	return _extract_frames_ffmpeg_cli(video_path, output_dir, frame_skip, max_frames)


def extract_frames(
	video_path: Path,
	output_dir: Path,
	frame_skip: int = 1,
	max_frames: int | None = None,
	backend: str = "auto",
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

	if backend not in {"auto", "gstreamer", "ffmpeg"}:
		raise ValueError(f"Unsupported backend: {backend}")

	if backend == "gstreamer":
		extracted = _extract_frames_gstreamer(video_path, output_dir, frame_skip, max_frames)
		if extracted == 0:
			raise RuntimeError(
				"GStreamer capture opened the video but did not decode any frames. Check the pipeline, plugins, "
				"and whether the input codec is supported by your OpenCV build."
			)
		print(f"Extraction complete: {extracted} frames saved to {output_dir}")
		return extracted

	if backend == "ffmpeg":
		extracted = _extract_frames_ffmpeg_cli(video_path, output_dir, frame_skip, max_frames)
		if extracted == 0:
			raise RuntimeError(
				"ffmpeg decoded no frames. Check that the input file is readable and contains a supported video stream."
			)
		print(f"Extraction complete: {extracted} frames saved to {output_dir}")
		return extracted

	extracted = _extract_frames_gstreamer_first(video_path, output_dir, frame_skip, max_frames)
	if extracted == 0:
		raise RuntimeError(
			"No frames were decoded. This commonly happens when the input codec is not supported by OpenCV. "
			"Install ffmpeg CLI or re-encode the source video to H.264/H.265 first."
		)
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
	parser.add_argument(
		"--backend",
		choices=("auto", "gstreamer", "ffmpeg"),
		default="auto",
		help=(
			"Video backend to use (default: auto; tries GStreamer first and falls back to ffmpeg CLI. "
			"Use ffmpeg to force the ffmpeg CLI path. GStreamer requires OpenCV built with GStreamer support.)"
		),
	)

	args = parser.parse_args()
	video_path = Path(args.video)
	output_dir = Path(args.output_dir)

	extract_frames(
		video_path=video_path,
		output_dir=output_dir,
		frame_skip=args.frame_skip,
		max_frames=args.max_frames,
		backend=args.backend,
	)


if __name__ == "__main__":
	main()
