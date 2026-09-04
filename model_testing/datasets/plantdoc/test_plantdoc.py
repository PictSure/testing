#!/usr/bin/env python3
"""Fine-grained tomato leaf disease classification, 1 and 3 shots.

Eight classes that differ only in lesion texture and color - the hardest task
in the suite, and the one where low double-digit accuracy is the expected
outcome rather than a bug.

    python3 model_testing/datasets/plantdoc/download.py
    python3 model_testing/datasets/plantdoc/test_plantdoc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def build_tasks() -> list[FixedTask]:
    return pooled_from_dir(
        DATA_DIR, name="PlantDoc", shot_candidates=(1, 3), max_queries_per_class=5
    ).tasks()


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
