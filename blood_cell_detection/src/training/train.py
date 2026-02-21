"""
src/training/train.py
Trains a YOLO11 model on the blood cell detection dataset.
"""

import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict) -> None:
    weights_dir = Path(config["model"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config["model"]["pretrained_weights"])
    print(f"Model loaded: {config['model']['pretrained_weights']}")

    results = model.train(
        data=config["data"]["yaml"],
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["imgsz"],
        batch=config["training"]["batch"],
        patience=config["training"]["patience"],
        lr0=config["training"]["lr0"],
        lrf=config["training"]["lrf"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
        warmup_epochs=config["training"]["warmup_epochs"],
        device=config["training"]["device"],
        project=config["model"]["experiments_dir"],
        name="run",
        exist_ok=False,
        save=True,
        plots=True,
        # Augmentation
        hsv_h=config["augmentation"]["hsv_h"],
        hsv_s=config["augmentation"]["hsv_s"],
        hsv_v=config["augmentation"]["hsv_v"],
        flipud=config["augmentation"]["flipud"],
        fliplr=config["augmentation"]["fliplr"],
        mosaic=config["augmentation"]["mosaic"],
        mixup=config["augmentation"]["mixup"],
    )

    # Save best weights to models/weights/
    best_weights_src = Path(config["model"]["experiments_dir"]) / "run" / "weights" / "best.pt"
    best_weights_dst = weights_dir / "best.pt"

    if best_weights_src.exists():
        import shutil
        shutil.copy(best_weights_src, best_weights_dst)
        print(f"Best weights saved to: {best_weights_dst}")

    print("Training complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
