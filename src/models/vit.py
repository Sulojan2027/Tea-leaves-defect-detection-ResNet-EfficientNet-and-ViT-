"""Pretrained Vision Transformer (ViT) fine-tuned for tea-leaf classification.

The backbone is loaded with KerasHub from a preset on the Hugging Face Hub
(https://huggingface.co/keras), so it is a native Keras model and trains,
saves and loads exactly like the CNNs.
"""

from __future__ import annotations

import keras
import keras_hub  # noqa: F401  (also registers the backbone class for keras.models.load_model)
from keras import layers

from src.config import ViTConfig


def build_vit(cfg: ViTConfig, num_classes: int, augmentation=None) -> keras.Model:
    """ViT backbone + [CLS]-token classification head.

    Takes raw [0, 255] pixels like the CNN models; scaling happens inside.
    """
    backbone = keras_hub.models.Backbone.from_preset(cfg.preset)
    preset_size = backbone.input_shape[1]
    if preset_size != cfg.image_size:
        raise ValueError(
            f"Preset {cfg.preset} expects {preset_size}px images, got image_size={cfg.image_size}"
        )
    backbone.trainable = not cfg.freeze_backbone

    inputs = keras.Input(shape=(cfg.image_size, cfg.image_size, 3))
    x = augmentation(inputs) if augmentation is not None else inputs
    # Same scaling as the preset's ViTImageConverter (mean=std=0.5): [0, 255] -> [-1, 1].
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="vit_preprocess")(x)

    tokens = backbone(x)  # (batch, 1 + num_patches, hidden_dim)
    features = tokens[:, 0, :]  # [CLS] token
    if cfg.dropout > 0:
        features = layers.Dropout(cfg.dropout)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)
    return keras.Model(inputs, outputs, name="vit")


def build_optimizer(cfg: ViTConfig, learning_rate: float | None = None) -> keras.optimizers.Optimizer:
    return keras.optimizers.AdamW(
        learning_rate=learning_rate or cfg.learning_rate, weight_decay=cfg.weight_decay
    )
