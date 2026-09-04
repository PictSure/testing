#!/usr/bin/env python3
"""Ten-way Sentinel-2 land-cover classification, 1 and 5 shots.

Overhead imagery has no canonical object orientation, so this probes whether
the encoders carry anything useful for a non-photographic viewpoint.

    python3 model_testing/datasets/eurosat/download.py
    python3 model_testing/datasets/eurosat/test_eurosat.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
NAME = "EuroSAT"

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
