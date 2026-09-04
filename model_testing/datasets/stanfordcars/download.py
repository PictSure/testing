#!/usr/bin/env python3
"""Stanford Cars - six car models (extremely fine-grained).

Six models from six different makes. Cars share an overall silhouette, so the
only discriminative signal is in grille, badge and body-line detail.

Only the handful of images the few-shot tasks need are fetched, via the Hugging
Face dataset viewer API.

    python3 model_testing/datasets/stanfordcars/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download_class_subset  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

DATASET = "tanganke/stanford_cars"
CONFIG = "default"
SPLIT = "test"
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# Images per class: enough for the largest context in test_stanfordcars.py plus its
# held-out queries.
PER_CLASS = 8

CLASSES = [
    "Acura TL Sedan 2012", "Audi S5 Coupe 2012", "BMW 3 Series Sedan 2012",
    "Bentley Continental GT Coupe 2012", "Bugatti Veyron 16.4 Coupe 2009",
    "Cadillac CTS-V Sedan 2012"
]

# Upstream class names are unwieldy as directory names; map them to slugs.
FOLDER_NAMES = {
    "Acura TL Sedan 2012": "acura_tl_sedan_2012",
    "Audi S5 Coupe 2012": "audi_s5_coupe_2012",
    "BMW 3 Series Sedan 2012": "bmw_3_series_sedan_2012",
    "Bentley Continental GT Coupe 2012": "bentley_continental_gt_coupe_2012",
    "Bugatti Veyron 16.4 Coupe 2009": "bugatti_veyron_coupe_2009",
    "Cadillac CTS-V Sedan 2012": "cadillac_cts_v_sedan_2012",
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
