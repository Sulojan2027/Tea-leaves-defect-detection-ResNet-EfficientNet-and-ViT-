"""Central configuration for data loading, training and models."""

import os
from dataclasses import dataclass
from pathlib import Path

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
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    augment: str = "none"
    early_stopping_patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-10
    output_dir: Path = Path("outputs")


@dataclass
class CNNConfig:
    head: str = "flatten"
    dense_units: int = 512
    dropout: float = 0.0


@dataclass
class ViTConfig:
    preset: str = "hf://keras/vit_base_patch16_224_imagenet"
    image_size: int = 224
    freeze_backbone: bool = False
    dropout: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    epochs: int = 20
