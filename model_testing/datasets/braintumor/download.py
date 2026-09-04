#!/usr/bin/env python3
"""BrainTumor_preprocessed - 4-way MRI tumor classification.

20 preprocessed MRI slices per class from PictSure/pictsure-library, enough to
run the same dataset at 1, 3, 5 and 10 shots.

    python3 model_testing/datasets/braintumor/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = (
    "https://raw.githubusercontent.com/PictSure/pictsure-library/main/Examples"
    "/BrainTumor_preprocessed"
)

# The upstream filenames use a two-letter class prefix that is not always the
# first two letters of the class name, so spell the mapping out.
CLASS_PREFIXES = {"glioma": "gl", "meningioma": "me", "notumor": "no", "pituitary": "pi"}
IMAGE_RANGE = range(10, 30)


def main() -> None:
    print(f"BrainTumor_preprocessed -> {DATA_DIR}")
    for cls, prefix in CLASS_PREFIXES.items():
        for i in IMAGE_RANGE:
            name = f"Te-{prefix}_00{i:02d}.jpg"
            download(f"{BASE_URL}/{cls}/{name}", DATA_DIR / cls / name)


if __name__ == "__main__":
    main()
