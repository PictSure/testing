#!/usr/bin/env python3
"""CatsDogs - the exact 2-shot cat/dog example from every PictSure model card.

Four context images (two cats, two dogs) plus one query, straight from
PictSure/pictsure-library. Tiny by design: this is the doc example, not a
benchmark.

    python3 model_testing/datasets/catsdogs/download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://raw.githubusercontent.com/PictSure/pictsure-library/main/Examples/CatsDogs"
FILES = ["cat1.jpg", "cat2.jpg", "dog1.jpg", "dog2.jpg", "query.jpg"]


def main() -> None:
    print(f"CatsDogs -> {DATA_DIR}")
    for name in FILES:
        download(f"{BASE_URL}/{name}", DATA_DIR / name)


if __name__ == "__main__":
    main()
