"""
src/inference/predict.py
Runs inference on images or video using the trained YOLO11 model.
Supports single image, folder of images, and video files.
"""

import yaml
import argparse
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def predict_images(
    model: YOLO,
    source: str,
    config: dict,
) -> list:
    """
    Runs inference on a single image or folder of images.

    Returns:
        List of dicts with per-image cell counts.
    """
    output_dir = Path(config["outputs"]["predictions"]) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source,
        conf=config["inference"]["conf"],
        iou=config["inference"]["iou"],
        save=config["inference"]["save"],
        project=str(output_dir),
        name="run",
        exist_ok=True,
    )

    all_counts = []
    for result in results:
        boxes = result.boxes
        classes = boxes.cls.tolist()
        names = result.names
        counts = Counter([names[int(c)] for c in classes])
        all_counts.append({
            "file": Path(result.path).name,
            "counts": dict(counts),
            "total": len(classes),
        })
        print(f"{Path(result.path).name} -> {dict(counts)}")

    return all_counts


def predict_video(
    model: YOLO,
    video_path: str,
    config: dict,
) -> None:
    """
    Runs inference on a video file and saves annotated output.
    """
    output_dir = Path(config["outputs"]["predictions"]) / "video"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on video: {video_path}")
    model.predict(
        source=video_path,
        conf=config["inference"]["conf"],
        iou=config["inference"]["iou"],
        save=True,
        project=str(output_dir),
        name="run",
        exist_ok=True,
        stream=True,   # Memory-efficient for long videos
    )
    print(f"Annotated video saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--source",  required=True, help="Image, folder, or video path")
    parser.add_argument("--weights", default=None,  help="Path to model weights")
    parser.add_argument("--mode",    default="image", choices=["image", "video"])
    args = parser.parse_args()

    config = load_config(args.config)

    weights = args.weights or str(
        Path(config["model"]["weights_dir"]) / "best.pt"
    )
    model = YOLO(weights)

    if args.mode == "video":
        predict_video(model, args.source, config)
    else:
        counts = predict_images(model, args.source, config)
        print(f"\nTotal images processed: {len(counts)}")
