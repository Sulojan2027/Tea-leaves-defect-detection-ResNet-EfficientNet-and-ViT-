"""Transfer-learning CNN classifiers (EfficientNetV2B3, ResNet50V2)."""

import keras
from keras import layers
from keras.applications import EfficientNetV2B3, ResNet50V2

from src.config import CNNConfig


def _classification_head(x, num_classes: int, cfg: CNNConfig):
    if cfg.head == "flatten":
        x = layers.Flatten()(x)
    elif cfg.head == "gap":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        raise ValueError(f"Unknown head '{cfg.head}', expected 'flatten' or 'gap'")
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    if cfg.dropout > 0:
        x = layers.Dropout(cfg.dropout)(x)
    return layers.Dense(num_classes, activation="softmax")(x)


def _build_transfer_model(base_cls, preprocess, name, image_size, num_classes, cfg, augmentation):
    input_shape = (image_size, image_size, 3)
    inputs = keras.Input(shape=input_shape)
    x = augmentation(inputs) if augmentation is not None else inputs
    if preprocess is not None:
        x = preprocess(x)

    base = base_cls(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    # training=False keeps BatchNorm in inference mode, also when fine-tuning later.
    x = base(x, training=False)

    outputs = _classification_head(x, num_classes, cfg)
    return keras.Model(inputs, outputs, name=name)


def build_efficientnet(image_size, num_classes, cfg: CNNConfig, augmentation=None):
    # EfficientNetV2 includes its own Rescaling layer and expects raw [0, 255] pixels.
    return _build_transfer_model(
        EfficientNetV2B3, None, "efficientnetv2b3", image_size, num_classes, cfg, augmentation
    )


def build_resnet(image_size, num_classes, cfg: CNNConfig, augmentation=None):
    # ResNet50V2 ImageNet weights expect pixels scaled to [-1, 1]
    # (same as keras.applications.resnet_v2.preprocess_input).
    preprocess = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="resnet_v2_preprocess")
    return _build_transfer_model(
        ResNet50V2, preprocess, "resnet50v2", image_size, num_classes, cfg, augmentation
    )


CNN_BUILDERS = {
    "efficientnet": build_efficientnet,
    "resnet": build_resnet,
}
