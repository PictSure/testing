#!/usr/bin/env python3
"""CIFAR-10 - ten coarse object categories at 32x32 pixels.

The low-resolution stress test of the suite: every encoder here expects a much
larger input, so these images are heavily upsampled before they reach it.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/cifar10/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "uoft-cs/cifar10"
CONFIG = "plain_text"
SPLIT = "test"
IMAGE_COLUMN = "img"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_cifar10.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]


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
    )


if __name__ == "__main__":
    main()
