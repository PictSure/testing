#!/usr/bin/env python3
"""Fashion-MNIST - ten clothing categories, 28x28 grayscale.

The most out-of-domain input in the suite: tiny, grayscale, and centered on a
black background, unlike anything the encoders saw during pretraining.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/fashionmnist/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "zalando-datasets/fashion_mnist"
CONFIG = "fashion_mnist"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_fashionmnist.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "T - shirt / top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

# Upstream class names are unwieldy as directory names; map them to slugs.
FOLDER_NAMES = {
    "T - shirt / top": "tshirt_top",
    "Trouser": "trouser",
    "Pullover": "pullover",
    "Dress": "dress",
    "Coat": "coat",
    "Sandal": "sandal",
    "Shirt": "shirt",
    "Sneaker": "sneaker",
    "Bag": "bag",
    "Ankle boot": "ankle_boot",
}


def main() -> None:
    download_class_subset(
        DATASET,
        DATA_DIR,
        CLASSES,
        PER_CLASS,
        config=CONFIG,
        split=SPLIT,
        image_column=IMAGE_COLUMN,
        label_column=LABEL_COLUMN,
        folder_names=FOLDER_NAMES,
    )


if __name__ == "__main__":
    main()
