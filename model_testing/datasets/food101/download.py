#!/usr/bin/env python3
"""Food-101 - eight dish categories (fine-grained food recognition).

Restricted to the first eight classes by label index: this dataset has no
viewer filter index, so the download pages through rows and buckets by label,
and low label indices keep that scan short.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/food101/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "ethz/food101"
CONFIG = "default"
SPLIT = "validation"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_food101.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
    "beignets", "bibimbap"
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
