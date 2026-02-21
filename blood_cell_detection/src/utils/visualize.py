"""
src/utils/visualize.py
Utility functions for visualizing predictions and dataset statistics.
"""

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Color map per class (BGR for OpenCV)
CLASS_COLORS = {
    "RBC":      (50,  50,  255),   # Red
    "WBC":      (50,  255, 50),    # Green
    "Platelets": (255, 150, 50),   # Orange
}


def draw_predictions(image_path: str, result, output_path: str = None) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image from a YOLO result.
    """
    img = cv2.imread(image_path)
    boxes = result.boxes
    names = result.names

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = names[int(box.cls)]
        conf = float(box.conf)
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if output_path:
        cv2.imwrite(output_path, img)

    return img


def plot_class_distribution(labels_dir: str, config: dict, save_path: str = None) -> None:
    """
    Plots class distribution from YOLO label files.
    """
    class_names = config["classes"]
    counts = Counter()

    for label_file in Path(labels_dir).glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                cls_idx = int(line.strip().split()[0])
                if cls_idx < len(class_names):
                    counts[class_names[cls_idx]] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.keys(), counts.values(),
           color=["#e74c3c", "#2ecc71", "#f39c12"])
    ax.set_title("Class Distribution in Dataset")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    for bar, val in zip(ax.patches, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(val),
                ha="center", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_summary(report_path: str, save_path: str = None) -> None:
    """
    Plots a bar chart of evaluation metrics from a JSON report.
    """
    import json

    with open(report_path, "r") as f:
        report = json.load(f)

    metrics = report["metrics"]
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#3498db", "#9b59b6", "#1abc9c", "#e67e22"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Evaluation Metrics")
    ax.set_ylabel("Score")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{val:.4f}",
                ha="center", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Metrics plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    config = load_config()
    viz_dir = Path(config["outputs"]["visualizations"])
    viz_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        labels_dir=config["data"]["processed"]["train_labels"],
        config=config,
        save_path=str(viz_dir / "class_distribution.png"),
    )
