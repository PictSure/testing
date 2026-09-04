#!/usr/bin/env python3
"""Runs every dataset's few-shot tasks through every PictSure model.

This is the "all at once" entry point. Each dataset under `datasets/` also has
its own runnable test script, and this collects all of them so each model is
loaded once and reused across every task:

    ./model_testing/download_datasets.sh     # all datasets
    python3 model_testing/test_models.py     # all tasks, all models

Useful variations:

    python3 model_testing/test_models.py --list
    python3 model_testing/test_models.py --datasets beans dtd gtsrb
    python3 model_testing/test_models.py --models pictsure/pictsure-clip

Every model repo is public, so no Hugging Face token is required; providing one
via HF_TOKEN (or `huggingface-cli login`) only avoids anonymous rate limits.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import FixedTask, MODEL_IDS, report, run_suite  # noqa: E402
from harness.registry import test_modules  # noqa: E402
from harness.runner import check_tasks  # noqa: E402


def collect_tasks(only: list[str] | None = None) -> tuple[list[FixedTask], list[str]]:
    tasks, skipped = [], []
    for slug, module in test_modules():
        if only and slug not in only:
            continue
        dataset_tasks = check_tasks(module.build_tasks())
        if not dataset_tasks:
            skipped.append(slug)
            continue
        tasks += dataset_tasks
    return tasks, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", metavar="SLUG",
        help="only run these dataset directories (default: all of them)",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODEL_IDS, metavar="ID",
        help="model ids to test (default: all five PictSure encoders)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the tasks that would run, then exit"
    )
    args = parser.parse_args()

    known = [slug for slug, _ in test_modules()]
    if args.datasets:
        unknown = [d for d in args.datasets if d not in known]
        if unknown:
            parser.error(f"unknown dataset(s) {unknown}; available: {known}")

    tasks, skipped = collect_tasks(args.datasets)
    if skipped:
        print(
            f"Skipped {len(skipped)} dataset(s) with no local images: {skipped}\n",
            file=sys.stderr,
        )
    if not tasks:
        print(
            "No runnable tasks - run ./model_testing/download_datasets.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list:
        for task in tasks:
            print(f"{task.name}: {len(task.context)} context, {len(task.queries)} queries")
        count = len(args.datasets or known)
        print(f"\n{len(tasks)} task(s) across {count} dataset(s)")
        return

    print(f"Running {len(tasks)} task(s) x {len(args.models)} model(s)\n")
    results = run_suite(tasks, args.models)
    sys.exit(0 if report(tasks, args.models, results) else 1)


if __name__ == "__main__":
    main()
