"""Evaluate a trained model on the validation split.

    python -m src.evaluate --run-dir outputs/efficientnet
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import src.models  # noqa: F401  (registers the custom ViT layers for loading)
from src.config import DataConfig
from src.data import load_datasets
from src.visualize import plot_confusion_matrix, plot_sample_predictions


def collect_predictions(model, dataset):
    """Predict batch by batch so labels and predictions are guaranteed aligned."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        y_prob.append(np.asarray(model.predict_on_batch(images)))
        y_true.append(np.argmax(labels.numpy(), axis=1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def compute_metrics(y_true, y_pred, class_names):
    # Micro-F1 equals accuracy for single-label classification, so report
    # macro/weighted F1, which actually reflect per-class performance.
    labels = list(range(len(class_names)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }


def evaluate(model, val_ds, class_names, out_dir: Path, title: str = ""):
    """Compute metrics and save metrics.json, confusion_matrix.png and sample_predictions.png."""
    y_true, y_prob = collect_predictions(model, val_ds)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png",
                          title=f"Confusion Matrix {title}".strip())
    plot_sample_predictions(model, val_ds, class_names, out_dir / "sample_predictions.png")

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run on the validation split")
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/efficientnet")
    parser.add_argument("--data-dir", type=Path, help="override the data_dir saved in run_config.json")
    args = parser.parse_args()

    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    data = run_config["data"]
    data_cfg = DataConfig(
        data_dir=args.data_dir or Path(data["data_dir"]),
        image_size=data["image_size"],
        batch_size=data["batch_size"],
        validation_split=data["validation_split"],
        seed=data["seed"],
    )
    _, val_ds, class_names = load_datasets(data_cfg)
    if class_names != run_config["class_names"]:
        raise ValueError(f"Class folders {class_names} do not match training {run_config['class_names']}")

    model = keras.models.load_model(args.run_dir / "model.keras")
    evaluate(model, val_ds, class_names, args.run_dir, title=run_config["model"])


if __name__ == "__main__":
    main()
