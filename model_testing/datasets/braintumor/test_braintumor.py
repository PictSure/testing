#!/usr/bin/env python3
"""4-way MRI tumor classification at 1, 3, 5 and 10 shots.

The widest shot sweep in the suite: 20 images per class leave room for a
10-shot context and still hold out queries, which is where in-context learning
should visibly improve.

    python3 model_testing/datasets/braintumor/download.py
    python3 model_testing/datasets/braintumor/test_braintumor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_tasks() -> list[FixedTask]:
    return pooled_from_dir(
        DATA_DIR,
        name="BrainTumor",
        classes=CLASSES,
        shot_candidates=(1, 3, 5, 10),
        max_queries_per_class=5,
    ).tasks()


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
