"""Dataset loading and augmentation."""

import keras
import tensorflow as tf
from keras import layers

from src.config import DataConfig

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(cfg: DataConfig):
    """Load the train/validation split from a class-per-folder directory.

    Returns (train_ds, val_ds, class_names). Labels are one-hot encoded.
    """
    if not cfg.data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at '{cfg.data_dir}'. Run `python -m src.download_data` "
            "or pass --data-dir / set TEA_DATA_DIR."
        )
    # subset="both" guarantees the two splits are disjoint for a given seed.
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        cfg.data_dir,
        label_mode="categorical",
        validation_split=cfg.validation_split,
        subset="both",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        image_size=(cfg.image_size, cfg.image_size),
    )
    class_names = train_ds.class_names

    # Shuffle individual samples after caching so every epoch sees a new order
    # (shuffling cached batches only reorders whole batches).
    train_ds = (
        train_ds.unbatch()
        .cache()
        .shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_augmentation() -> keras.Sequential:
    """Random augmentations on raw [0, 255] pixels, active only during training."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            layers.RandomContrast(factor=0.2),
        ],
        name="data_augmentation",
    )
