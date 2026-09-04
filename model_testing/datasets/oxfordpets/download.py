#!/usr/bin/env python3
"""Oxford-IIIT Pet - eight cat and dog breeds.

A fine-grained sibling of the CatsDogs doc example: same animals, but the
model now has to tell breeds apart rather than cats from dogs.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/oxfordpets/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "timm/oxford-iiit-pet"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_oxfordpets.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "abyssinian", "bengal", "persian", "siamese", "beagle", "boxer", "chihuahua", "pug"
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
