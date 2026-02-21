"""
run_pipeline.py
Master script that runs the full blood cell detection pipeline:
  1. Download dataset
  2. (Optional) Extract video frames
  3. Train YOLO11 model
  4. Evaluate on test set
  5. Run inference
  6. Generate visualizations
"""

import argparse
import yaml
from pathlib import Path

from src.data.download_dataset import download_dataset, organize_dataset, validate_dataset
from src.data.extract_video_frames import extract_frames
from src.training.train import train
from src.evaluation.evaluate import evaluate
from src.inference.predict import predict_images, predict_video
from src.utils.visualize import plot_class_distribution, plot_metrics_summary
from ultralytics import YOLO


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config: dict,
    steps: list = None,
    inference_source: str = None,
) -> None:

    if steps is None:
        steps = ["download", "train", "evaluate", "visualize"]

    print("\n" + "="*60)
    print("   BLOOD CELL DETECTION — FULL PIPELINE")
    print("="*60 + "\n")

    # ------------------------------------------------------------------ #
    # STEP 1: Download Dataset
    # ------------------------------------------------------------------ #
    if "download" in steps:
        print("[1/6] Downloading dataset from Roboflow...")
        dataset_location = download_dataset(config)
        organize_dataset(dataset_location, config)
        validate_dataset(config)
        print("Dataset ready.\n")

    # ------------------------------------------------------------------ #
    # STEP 2: Extract Video Frames (optional)
    # ------------------------------------------------------------------ #
    if "video" in steps:
        print("[2/6] Extracting video frames...")
        video_raw_dir = Path(config["data"]["video"]["raw"])
        frames_dir = config["data"]["video"]["frames"]
        for video_file in video_raw_dir.glob("*.mp4"):
            extract_frames(
                video_path=str(video_file),
                output_dir=frames_dir,
                frame_interval=5,
            )
        print("Video frames extracted.\n")

    # ------------------------------------------------------------------ #
    # STEP 3: Training
    # ------------------------------------------------------------------ #
    if "train" in steps:
        print("[3/6] Starting training...")
        train(config)
        print("Training complete.\n")

    # ------------------------------------------------------------------ #
    # STEP 4: Evaluation
    # ------------------------------------------------------------------ #
    if "evaluate" in steps:
        print("[4/6] Evaluating model on test set...")
        results = evaluate(config)
        print("Evaluation complete.\n")

    # ------------------------------------------------------------------ #
    # STEP 5: Inference
    # ------------------------------------------------------------------ #
    if "inference" in steps and inference_source:
        print(f"[5/6] Running inference on: {inference_source}")
        weights = str(Path(config["model"]["weights_dir"]) / "best.pt")
        model = YOLO(weights)
        source_path = Path(inference_source)
        if source_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
            predict_video(model, inference_source, config)
        else:
            predict_images(model, inference_source, config)
        print("Inference complete.\n")

    # ------------------------------------------------------------------ #
    # STEP 6: Visualizations
    # ------------------------------------------------------------------ #
    if "visualize" in steps:
        print("[6/6] Generating visualizations...")
        viz_dir = Path(config["outputs"]["visualizations"])
        viz_dir.mkdir(parents=True, exist_ok=True)

        plot_class_distribution(
            labels_dir=config["data"]["processed"]["train_labels"],
            config=config,
            save_path=str(viz_dir / "class_distribution.png"),
        )

        reports_dir = Path(config["outputs"]["reports"])
        report_files = sorted(reports_dir.glob("evaluation_*.json"))
        if report_files:
            plot_metrics_summary(
                report_path=str(report_files[-1]),
                save_path=str(viz_dir / "metrics_summary.png"),
            )

        print("Visualizations saved.\n")

    print("="*60)
    print("   PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood Cell Detection Pipeline")
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--steps",    nargs="+",
                        default=["download", "train", "evaluate", "visualize"],
                        choices=["download", "video", "train", "evaluate", "inference", "visualize"],
                        help="Pipeline steps to run")
    parser.add_argument("--source",   default=None,
                        help="Source for inference (image path, folder, or video)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, steps=args.steps, inference_source=args.source)
