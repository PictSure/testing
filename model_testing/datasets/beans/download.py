#!/usr/bin/env python3
"""Beans - three-way bean leaf disease classification (Makerere AI Lab).

A small, clean agricultural dataset: two diseases plus healthy leaves, close
enough in appearance to be non-trivial but far from PlantDoc's difficulty.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/beans/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "AI-Lab-Makerere/beans"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "labels"

# Images per class: enough for the largest context in test_beans.py plus its
# held-out queries.
PER_CLASS = 12

CLASSES = [
    "angular_leaf_spot", "bean_rust", "healthy"
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
