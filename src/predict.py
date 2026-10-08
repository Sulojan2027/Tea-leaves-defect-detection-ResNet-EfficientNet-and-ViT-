"""Predict the disease class of new leaf images with a trained run."""

import argparse
import json
from pathlib import Path

import keras
import numpy as np

import src.models


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = keras.utils.load_img(path, target_size=(image_size, image_size), interpolation="bilinear")
    return keras.utils.img_to_array(img)


def main():
    parser = argparse.ArgumentParser(description="Predict tea-leaf disease for images")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/efficientnet")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    run_config = json.loads((args.run_dir / "run_config.json").read_text())
    class_names = run_config["class_names"]
    image_size = run_config["data"]["image_size"]
    model = keras.models.load_model(args.run_dir / "model.keras")

    batch = np.stack([load_image(p, image_size) for p in args.images])
    probs = model.predict(batch, verbose=0)

    for path, p in zip(args.images, probs):
        top = np.argsort(p)[::-1][: args.top_k]
        ranked = ", ".join(f"{class_names[i]} ({p[i]:.1%})" for i in top)
        print(f"{path}: {ranked}")


if __name__ == "__main__":
    main()
