#!/usr/bin/env python3
"""DTD (Describable Textures) - eight texture categories.

Textures have no object to latch onto, which separates encoders that learned
general visual statistics from ones that learned object semantics.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/dtd/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "tanganke/dtd"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_dtd.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "banded", "bubbly", "chequered", "cobwebbed", "honeycombed", "marbled", "striped",
    "zigzagged"
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
