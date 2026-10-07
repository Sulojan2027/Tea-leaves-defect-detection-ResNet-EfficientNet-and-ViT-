"""Download the Kaggle tea-leaf dataset and link it to data/tea-leaves.

    python -m src.download_data

kagglehub caches the files under ~/.cache/kagglehub; this script then points
a symlink at the folder that holds the class sub-folders, so the training
scripts' default --data-dir works without copying the images.
"""

import argparse
from pathlib import Path

import kagglehub

from src.config import DEFAULT_DATA_DIR, KAGGLE_DATASET

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_root(root: Path) -> Path:
    """Return the first directory whose sub-folders all contain images."""
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        if len(subdirs) >= 2 and all(
            any(f.suffix.lower() in IMAGE_EXTS for f in s.iterdir()) for s in subdirs
        ):
            return d
    raise FileNotFoundError(f"No class-per-folder image directory found under {root}")


def main():
    parser = argparse.ArgumentParser(description="Download the tea-leaf dataset from Kaggle")
    parser.add_argument("--link", type=Path, default=DEFAULT_DATA_DIR, help="symlink to create")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    download_root = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=args.force))
    class_root = find_class_root(download_root)

    args.link.parent.mkdir(parents=True, exist_ok=True)
    if args.link.is_symlink():
        args.link.unlink()
    elif args.link.exists():
        raise FileExistsError(f"{args.link} exists and is not a symlink; remove it or pass --link")
    args.link.symlink_to(class_root.resolve(), target_is_directory=True)

    print(f"Dataset: {class_root}")
    print(f"Linked:  {args.link} -> {class_root}")
    for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        print(f"  {class_dir.name:<16} {n} images")


if __name__ == "__main__":
    main()
