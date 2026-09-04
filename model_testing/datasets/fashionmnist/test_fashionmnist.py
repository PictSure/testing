#!/usr/bin/env python3
"""Ten clothing categories from 28x28 grayscale thumbnails, 1 and 5 shots.

Deliberately adversarial for pretrained photographic encoders. Several classes
(Pullover / Coat / Shirt) are near-indistinguishable at this resolution.

    python3 model_testing/datasets/fashionmnist/download.py
    python3 model_testing/datasets/fashionmnist/test_fashionmnist.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "FashionMNIST"

SHOT_CANDIDATES = (1, 5)
MAX_QUERIES_PER_CLASS = 3


def build_tasks() -> list[FixedTask]:
    return pooled_from_dir(
        DATA_DIR,
        name=NAME,
        shot_candidates=SHOT_CANDIDATES,
        max_queries_per_class=MAX_QUERIES_PER_CLASS,
    ).tasks()


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
