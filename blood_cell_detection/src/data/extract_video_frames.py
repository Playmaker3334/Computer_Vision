"""
src/data/extract_video_frames.py
Extracts frames from video files for inference or training augmentation.
"""

import cv2
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
) -> int:
    """
    Extracts frames from a video file.

    Args:
        video_path:     Path to input video file (.mp4, .avi, etc.)
        output_dir:     Directory to save extracted frames
        frame_interval: Save every N frames (1 = all frames)

    Returns:
        Total number of frames saved
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames} | FPS: {fps:.2f}")

    saved = 0
    frame_idx = 0
    video_name = Path(video_path).stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(Path(output_dir) / filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Frames saved: {saved} -> {output_dir}")
    return saved


if __name__ == "__main__":
    config = load_config()
    video_raw_dir = Path(config["data"]["video"]["raw"])
    frames_dir = config["data"]["video"]["frames"]

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = [
        f for f in video_raw_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {video_raw_dir}")
    else:
        for video_file in video_files:
            extract_frames(
                video_path=str(video_file),
                output_dir=frames_dir,
                frame_interval=5,   # Save every 5th frame
            )
