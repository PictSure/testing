#!/usr/bin/env python3
"""Eight traffic sign classes at 1, 3 and 5 shots.

Three of the eight are speed-limit signs distinguishable only by their digits,
which is effectively an OCR task - a useful probe of whether the encoders
retain any text sensitivity (CLIP in particular).

    python3 model_testing/datasets/gtsrb/download.py
    python3 model_testing/datasets/gtsrb/test_gtsrb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "GTSRB"

SHOT_CANDIDATES = (1, 3, 5)
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
