#!/usr/bin/env python3
"""The README's own cat/dog example, as a task.

Fixed context and a single query, so treat the result as a smoke test that the
documented call sequence works - not as an accuracy measurement.

    python3 model_testing/datasets/catsdogs/download.py
    python3 model_testing/datasets/catsdogs/test_catsdogs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import FixedTask, cli  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"

# label 0 = cat, 1 = dog; query.jpg is a dog.
CONTEXT = [("cat1.jpg", 0), ("cat2.jpg", 0), ("dog1.jpg", 1), ("dog2.jpg", 1)]
QUERIES = [("query.jpg", 1)]


def build_tasks() -> list[FixedTask]:
    return [
        FixedTask(
            name="CatsDogs (README example, 2-shot)",
            context=[(DATA_DIR / n, label) for n, label in CONTEXT],
            queries=[(DATA_DIR / n, label) for n, label in QUERIES],
        )
    ]


if __name__ == "__main__":
    cli(build_tasks, __doc__.splitlines()[0], DATA_DIR)
