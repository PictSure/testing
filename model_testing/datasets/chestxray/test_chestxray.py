#!/usr/bin/env python3
"""Binary pneumonia detection from chest radiographs, 1 to 10 shots.

Two classes over grayscale X-rays: a domain none of the encoders were
pretrained on, which makes the shot-count trend the interesting signal here.

    python3 model_testing/datasets/chestxray/download.py
    python3 model_testing/datasets/chestxray/test_chestxray.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "ChestXray"

SHOT_CANDIDATES = (1, 3, 5, 10)
MAX_QUERIES_PER_CLASS = 5


def build_tasks() -> list[FixedTask]:
    return pooled_from_dir(
        DATA_DIR,
        name=NAME,
        shot_candidates=SHOT_CANDIDATES,
        max_queries_per_class=MAX_QUERIES_PER_CLASS,
    ).tasks()


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
