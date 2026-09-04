#!/usr/bin/env python3
"""EuroSAT (RGB) - ten Sentinel-2 land-cover classes.

Overhead satellite imagery at 64x64: a viewpoint and scale none of the
encoders were pretrained on, with all ten classes kept.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/eurosat/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "blanchon/EuroSAT_RGB"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_eurosat.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
    "Pasture", "Permanent Crop", "Residential Buildings", "River", "SeaLake"
]

# Upstream class names are unwieldy as directory names; map them to slugs.
FOLDER_NAMES = {
    "Annual Crop": "annual_crop",
    "Forest": "forest",
    "Herbaceous Vegetation": "herbaceous_vegetation",
    "Highway": "highway",
    "Industrial Buildings": "industrial_buildings",
    "Pasture": "pasture",
    "Permanent Crop": "permanent_crop",
    "Residential Buildings": "residential_buildings",
    "River": "river",
    "SeaLake": "sea_lake",
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
