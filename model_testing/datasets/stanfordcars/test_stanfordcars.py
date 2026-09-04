#!/usr/bin/env python3
"""Six car models at 1, 3 and 5 shots - the fine-grained ceiling test.

All six share a car silhouette; telling them apart needs grille and body-line
detail that survives few pretraining objectives. Expect PlantDoc-like numbers.

    python3 model_testing/datasets/stanfordcars/download.py
    python3 model_testing/datasets/stanfordcars/test_stanfordcars.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "StanfordCars"

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
