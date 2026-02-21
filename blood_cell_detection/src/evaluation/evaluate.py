"""
src/evaluation/evaluate.py
Evaluates the trained model on the test set and generates a metrics report.
"""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(config: dict, weights_path: str = None) -> dict:
    if weights_path is None:
        weights_path = str(Path(config["model"]["weights_dir"]) / "best.pt")

    print(f"Evaluating model: {weights_path}")
    model = YOLO(weights_path)

    metrics = model.val(
        data=config["data"]["yaml"],
        conf=config["evaluation"]["conf_threshold"],
        iou=config["evaluation"]["iou_threshold"],
        plots=config["evaluation"]["save_plots"],
        save_json=True,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "weights": weights_path,
        "metrics": {
            "mAP50":       round(float(metrics.box.map50), 4),
            "mAP50_95":    round(float(metrics.box.map), 4),
            "precision":   round(float(metrics.box.mp), 4),
            "recall":      round(float(metrics.box.mr), 4),
        },
        "per_class": {}
    }

    class_names = config["classes"]
    for i, name in enumerate(class_names):
        try:
            results["per_class"][name] = {
                "AP50": round(float(metrics.box.ap50[i]), 4),
                "AP":   round(float(metrics.box.ap[i]), 4),
            }
        except (IndexError, AttributeError):
            pass

    reports_dir = Path(config["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n--- Evaluation Results ---")
    print(f"mAP@50:    {results['metrics']['mAP50']}")
    print(f"mAP@50:95: {results['metrics']['mAP50_95']}")
    print(f"Precision: {results['metrics']['precision']}")
    print(f"Recall:    {results['metrics']['recall']}")
    print(f"\nPer-class results:")
    for cls, vals in results["per_class"].items():
        print(f"  {cls}: AP50={vals['AP50']} | AP={vals['AP']}")
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, weights_path=args.weights)
