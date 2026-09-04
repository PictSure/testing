#!/usr/bin/env python3
"""Chest X-ray pneumonia - binary NORMAL vs PNEUMONIA.

A second medical-imaging domain alongside BrainTumor, but grayscale
radiographs instead of MRI slices, and binary instead of 4-way.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/chestxray/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "hf-vision/chest-xray-pneumonia"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_chestxray.py plus its
# held-out queries.
PER_CLASS = 12

CLASSES = [
    "NORMAL", "PNEUMONIA"
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
