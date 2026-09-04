#!/usr/bin/env python3
"""Caltech-101 - eight classic object categories.

The canonical pre-ImageNet object recognition benchmark, subset to eight
unambiguous categories. Objects are large, centered and well lit.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/caltech101/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "dpdl-benchmark/caltech101"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_caltech101.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "accordion", "airplanes", "anchor", "butterfly", "dolphin", "elephant", "grand_piano",
    "laptop"
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
