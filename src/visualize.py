"""Plotting helpers. All functions save to `path` instead of calling plt.show()."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, class_names, path, title="Confusion Matrix", cmap="Greens"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(model, dataset, class_names, path, n=8):
    """Plot the first `n` images of `dataset` with predicted and true labels."""
    images, labels = next(iter(dataset.take(1)))
    n = min(n, images.shape[0])
    probs = model.predict_on_batch(images[:n])

    fig, axs = plt.subplots(1, n, figsize=(2.5 * n, 3.5))
    for i, ax in enumerate(np.atleast_1d(axs)):
        pred, true = int(np.argmax(probs[i])), int(np.argmax(labels[i]))
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(
            f"Pred: {class_names[pred]}\nTrue: {class_names[true]}",
            color="green" if pred == true else "red",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

