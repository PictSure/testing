#!/usr/bin/env python3
"""GTSRB - eight German traffic sign classes.

Small, low-resolution crops of real signs, including three speed-limit signs
that differ only in the digits printed on them.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/gtsrb/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "tanganke/gtsrb"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_gtsrb.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "red and white circle 30 kph speed limit", "red and white circle 50 kph speed limit",
    "red and white circle 80 kph speed limit", "stop",
    "white and yellow diamond priority road",
    "red and white upside down triangle yield right-of-way",
    "red circle with white horizonal stripe no entry",
    "blue circle with white keep right arrow mandatory"
]

# Upstream class names are unwieldy as directory names; map them to slugs.
FOLDER_NAMES = {
    "red and white circle 30 kph speed limit": "speed_30",
    "red and white circle 50 kph speed limit": "speed_50",
    "red and white circle 80 kph speed limit": "speed_80",
    "stop": "stop",
    "white and yellow diamond priority road": "priority_road",
    "red and white upside down triangle yield right-of-way": "yield",
    "red circle with white horizonal stripe no entry": "no_entry",
    "blue circle with white keep right arrow mandatory": "keep_right",
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
