"""Train a tea-leaf disease classifier and evaluate it on the validation split."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import keras

from src.config import DEFAULT_DATA_DIR, SEED, CNNConfig, DataConfig, TrainConfig, ViTConfig
from src.data import build_augmentation, load_datasets
from src.evaluate import evaluate
from src.models import CNN_BUILDERS, MODEL_NAMES
from src.models.vit import build_optimizer, build_vit


def build_callbacks(cfg: TrainConfig, run_dir: Path, resume: bool):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            str(ckpt_dir / "best.keras"), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(str(ckpt_dir / "last.keras")),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            verbose=1,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            verbose=1,
            min_lr=cfg.min_lr,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=resume),
    ]


def completed_epochs(run_dir: Path) -> int:
    history = run_dir / "history.csv"
    if not history.exists():
        return 0
    with history.open() as f:
        return sum(1 for _ in csv.DictReader(f))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--image-size", type=int, help="CNNs only (default 500); the ViT uses its preset's size")
    parser.add_argument("--batch-size", type=int, default=DataConfig.batch_size)
    parser.add_argument("--epochs", type=int, help="default: 30 for CNNs, 20 for ViT")
    parser.add_argument("--learning-rate", type=float, help="default: 1e-3 for CNNs, 2e-5 for ViT")
    parser.add_argument("--weight-decay", type=float, help="AdamW decay; default: 0 (plain Adam) for CNNs, 1e-4 for ViT")
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--lr-patience", type=int, default=TrainConfig.lr_patience)
    parser.add_argument("--head", choices=["flatten", "gap"], default=CNNConfig.head, help="CNN head")
    parser.add_argument("--dropout", type=float, help="head dropout; default: 0 for CNNs, 0.1 for ViT")
    parser.add_argument(
        "--augment", nargs="?", const="light", default=TrainConfig.augment,
        choices=["none", "light", "strong"], help="bare --augment means 'light'",
    )
    parser.add_argument("--freeze-backbone", action="store_true", help="ViT: train only the head")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/last.keras")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_model(model_name, data_cfg, train_cfg, cnn_cfg, vit_cfg, num_classes):
    augmentation = build_augmentation(train_cfg.augment)
    if model_name == "vit":
        model = build_vit(vit_cfg, num_classes, augmentation)
        optimizer = build_optimizer(vit_cfg)
    else:
        model = CNN_BUILDERS[model_name](data_cfg.image_size, num_classes, cnn_cfg, augmentation)
        if train_cfg.weight_decay > 0:
            optimizer = keras.optimizers.AdamW(
                learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
            )
        else:
            optimizer = keras.optimizers.Adam(learning_rate=train_cfg.learning_rate)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=train_cfg.label_smoothing)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    is_vit = args.model == "vit"

    vit_cfg = ViTConfig(freeze_backbone=args.freeze_backbone)
    cnn_cfg = CNNConfig(head=args.head)
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        label_smoothing=args.label_smoothing,
        augment=args.augment,
        lr_patience=args.lr_patience,
    )
    if is_vit:
        vit_cfg.epochs = args.epochs or vit_cfg.epochs
        vit_cfg.learning_rate = args.learning_rate or vit_cfg.learning_rate
        if args.weight_decay is not None:
            vit_cfg.weight_decay = args.weight_decay
        if args.dropout is not None:
            vit_cfg.dropout = args.dropout
        train_cfg.epochs = vit_cfg.epochs
        train_cfg.learning_rate = vit_cfg.learning_rate
        train_cfg.weight_decay = vit_cfg.weight_decay
    else:
        train_cfg.epochs = args.epochs or train_cfg.epochs
        train_cfg.learning_rate = args.learning_rate or train_cfg.learning_rate
        if args.weight_decay is not None:
            train_cfg.weight_decay = args.weight_decay
        if args.dropout is not None:
            cnn_cfg.dropout = args.dropout
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=vit_cfg.image_size if is_vit else (args.image_size or DataConfig.image_size),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_ds, val_ds, class_names = load_datasets(data_cfg)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    run_dir = train_cfg.output_dir / args.model
    last_ckpt = run_dir / "checkpoints" / "last.keras"
    initial_epoch = 0
    if args.resume:
        if not last_ckpt.exists():
            raise FileNotFoundError(f"--resume given but {last_ckpt} does not exist")
        model = keras.models.load_model(last_ckpt)
        initial_epoch = completed_epochs(run_dir)
        print(f"Resuming from {last_ckpt} at epoch {initial_epoch}")
    else:
        model = build_model(args.model, data_cfg, train_cfg, cnn_cfg, vit_cfg, num_classes)
    model.summary()

    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": args.model,
        "class_names": class_names,
        "data": asdict(data_cfg),
        "train": asdict(train_cfg),
        "model_config": asdict(vit_cfg) if is_vit else asdict(cnn_cfg),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    model.fit(
        train_ds,
        epochs=train_cfg.epochs,
        initial_epoch=initial_epoch,
        validation_data=val_ds,
        callbacks=build_callbacks(train_cfg, run_dir, resume=args.resume),
    )
    model.save(run_dir / "model.keras")
    evaluate(model, val_ds, class_names, run_dir, title=args.model)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
