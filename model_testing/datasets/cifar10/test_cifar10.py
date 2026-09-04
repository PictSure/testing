#!/usr/bin/env python3
"""Ten coarse object categories at 32x32, 1 and 5 shots.

Tests how the encoders cope with inputs far below their pretraining
resolution. Ten classes make each task expensive, so only two shot counts run.

    python3 model_testing/datasets/cifar10/download.py
    python3 model_testing/datasets/cifar10/test_cifar10.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "CIFAR10"

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
