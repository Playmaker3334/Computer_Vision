"""
src/data/download_dataset.py
Downloads the BCCD dataset from Roboflow in YOLOv8 format
and organizes it into the project structure.
"""

import os
import shutil
import yaml
from pathlib import Path
from roboflow import Roboflow


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(config: dict) -> str:
    """Download dataset from Roboflow and return dataset location."""
    rf = Roboflow(api_key=config["roboflow"]["api_key"])
    project = rf.workspace(config["roboflow"]["workspace"]).project(
        config["roboflow"]["project"]
    )
    dataset = project.version(config["roboflow"]["version"]).download(
        config["roboflow"]["format"],
        location="data/raw"
    )
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def organize_dataset(dataset_location: str, config: dict) -> None:
    """
    Moves downloaded dataset files into the processed folder structure
    expected by the project.
    """
    splits = ["train", "val", "test"]
    for split in splits:
        src_images = Path(dataset_location) / split / "images"
        src_labels = Path(dataset_location) / split / "labels"

        dst_images = Path(config["data"]["processed"][f"{split}_images"])
        dst_labels = Path(config["data"]["processed"][f"{split}_labels"])

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        if src_images.exists():
            for f in src_images.iterdir():
                shutil.copy(f, dst_images / f.name)

        if src_labels.exists():
            for f in src_labels.iterdir():
                shutil.copy(f, dst_labels / f.name)

        print(f"[{split}] images: {len(list(dst_images.iterdir()))} | "
              f"labels: {len(list(dst_labels.iterdir()))}")


def validate_dataset(config: dict) -> None:
    """Validates image/label count parity per split."""
    splits = ["train", "val", "test"]
    for split in splits:
        images = list(Path(config["data"]["processed"][f"{split}_images"]).iterdir())
        labels = list(Path(config["data"]["processed"][f"{split}_labels"]).iterdir())
        assert len(images) == len(labels), (
            f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels"
        )
        print(f"[{split}] Validated: {len(images)} image-label pairs")


if __name__ == "__main__":
    config = load_config()
    dataset_location = download_dataset(config)
    organize_dataset(dataset_location, config)
    validate_dataset(config)
    print("Dataset ready.")
