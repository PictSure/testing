#!/usr/bin/env python3
"""Five wildflower species at 1, 3 and 5 shots.

Visually distinct natural classes - the easy end of the suite, and a useful
control: an encoder that struggles here is struggling at the embedding level,
not at fine-grained discrimination.

    python3 model_testing/datasets/swedishflowers/download.py
    python3 model_testing/datasets/swedishflowers/test_swedishflowers.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli, pooled_from_dir  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def build_tasks() -> list[FixedTask]:
    return pooled_from_dir(
        DATA_DIR, name="SwedishFlowers", shot_candidates=(1, 3, 5), max_queries_per_class=5
    ).tasks()


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
