"""Central configuration for data loading, training and models."""

import os
from dataclasses import dataclass
from pathlib import Path

# Where `python -m src.download_data` links the dataset; override with the
# TEA_DATA_DIR env var or --data-dir (e.g. the /kaggle/input/... path on Kaggle).
DEFAULT_DATA_DIR = Path(os.environ.get("TEA_DATA_DIR", "data/tea-leaves"))
KAGGLE_DATASET = "shashwatwork/identifying-disease-in-tea-leafs"
SEED = 42


@dataclass
class DataConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    image_size: int = 500
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = SEED
    shuffle_buffer: int = 1024


@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3  # Keras Adam default, as used in the notebook
    early_stopping_patience: int = 10
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-10
    output_dir: Path = Path("outputs")


@dataclass
class CNNConfig:
    head: str = "flatten"  # "flatten" (notebook) or "gap" (GlobalAveragePooling)
    dense_units: int = 512
    dropout: float = 0.0
    augment: bool = False


@dataclass
class ViTConfig:
    # KerasHub preset hosted on the Hugging Face Hub. Other options:
    # vit_base_patch16_224_imagenet21k, vit_base_patch16_384_imagenet (image_size=384), ...
    preset: str = "hf://keras/vit_base_patch16_224_imagenet"
    image_size: int = 224  # must match the preset's input resolution
    freeze_backbone: bool = False  # True = train only the classification head
    dropout: float = 0.1
    # Full fine-tuning of a pretrained transformer needs a small learning rate.
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    epochs: int = 20
