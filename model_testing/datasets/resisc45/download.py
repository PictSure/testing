#!/usr/bin/env python3
"""RESISC45 - remote-sensing scene classification (8-class subset).

A second overhead-imagery dataset alongside EuroSAT, but at higher resolution
and with scene categories rather than land cover. Subset to eight classes to
keep task cost in line with the rest of the suite.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/resisc45/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "timm/resisc45"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_resisc45.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "airplane", "beach", "desert", "forest", "freeway", "harbor", "island", "stadium"
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
