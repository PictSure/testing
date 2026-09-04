"""Runs FixedTasks through every PictSure model and reports the results.

This is the part that actually exercises the documented usage pattern -
`PictSure.from_pretrained(...)` -> `set_context_images(...)` -> `predict(...)` -
so a README change can be validated against real model behavior.
"""
import argparse
import sys
import time
from pathlib import Path

from .net import resolve_hf_token
from .tasks import FixedTask

MODEL_IDS = [
    "pictsure/pictsure-vit",
    "pictsure/pictsure-resnet",
    "pictsure/pictsure-dinov2",
    "pictsure/pictsure-dinov2-large",
    "pictsure/pictsure-clip",
]


class TaskResult:
    def __init__(self, ok: bool, message: str, correct: int = 0, total: int = 0):
        self.ok = ok
        self.message = message
        self.correct = correct
        self.total = total

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.ok and self.total else None

    def cell(self) -> str:
        if not self.ok:
            return "FAIL"
        return f"{round(100 * self.accuracy)}% ({self.correct}/{self.total})"


def run_task(model, task: FixedTask) -> TaskResult:
    from PIL import Image

    try:
        context_images = [Image.open(p).convert("RGB") for p, _ in task.context]
        context_labels = [label for _, label in task.context]
        model.set_context_images(context_images, context_labels)

        correct = 0
        for path, expected in task.queries:
            prediction = model.predict(Image.open(path).convert("RGB"))
            correct += int(prediction == expected)
        total = len(task.queries)
        return TaskResult(True, f"{correct}/{total} correct", correct, total)
    except Exception as e:  # noqa: BLE001
        return TaskResult(False, f"{type(e).__name__}: {e}")


def check_tasks(tasks: list[FixedTask]) -> list[FixedTask]:
    """Drop tasks whose images are not on disk, loudly."""
    runnable = []
    for task in tasks:
        missing = task.missing_files()
        if missing:
            print(
                f"SKIP {task.name}: {len(missing)} image(s) missing "
                f"(e.g. {missing[0]}) - run the download script for this dataset",
                file=sys.stderr,
            )
            continue
        runnable.append(task)
    return runnable


def run_suite(tasks: list[FixedTask], model_ids: list[str] = MODEL_IDS) -> dict:
    from PictSure import PictSure

    token = resolve_hf_token()
    results: dict[tuple[str, str], TaskResult] = {}

    for model_id in model_ids:
        print(f"=== {model_id} ===", flush=True)
        try:
            model = PictSure.from_pretrained(model_id, token=token)
        except Exception as e:  # noqa: BLE001
            for task in tasks:
                results[(model_id, task.name)] = TaskResult(
                    False, f"load failed: {type(e).__name__}: {e}"
                )
            print(f"FAIL load: {type(e).__name__}: {e}", flush=True)
            continue

        for task in tasks:
            started = time.monotonic()
            result = run_task(model, task)
            results[(model_id, task.name)] = result
            flag = "OK  " if result.ok else "FAIL"
            elapsed = time.monotonic() - started
            print(f"{flag} {task.name}: {result.message} [{elapsed:.1f}s]", flush=True)
        print(flush=True)

    return results


def markdown_table(tasks: list[FixedTask], model_ids: list[str], results: dict) -> str:
    short = [m.removeprefix("pictsure/pictsure-") for m in model_ids]
    lines = ["| Task | " + " | ".join(short) + " |", "|---" * (len(short) + 1) + "|"]
    for task in tasks:
        row = [results.get((m, task.name)) for m in model_ids]
        best = max((r.accuracy for r in row if r and r.accuracy is not None), default=None)
        cells = []
        for r in row:
            text = r.cell() if r else "-"
            if r and best is not None and r.accuracy == best:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {task.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(tasks: list[FixedTask], model_ids: list[str], results: dict) -> bool:
    print("=== Summary ===")
    all_ok = True
    for (model_id, task_name), result in results.items():
        print(f"{'OK  ' if result.ok else 'FAIL'} {model_id} / {task_name}: {result.message}")
        all_ok = all_ok and result.ok
    print("\n=== Results table (markdown) ===")
    print(markdown_table(tasks, model_ids, results))
    return all_ok


def cli(build_tasks, description: str, data_dir: Path | None = None) -> None:
    """Entry point shared by the per-dataset test scripts and test_models.py."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--models", nargs="+", default=MODEL_IDS,
        help="model ids to test (default: all five PictSure encoders)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list the tasks that would run, then exit"
    )
    args = parser.parse_args()

    if data_dir is not None and not data_dir.exists():
        print(
            f"Data dir {data_dir} missing - run the matching download script first.",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks = check_tasks(build_tasks())
    if not tasks:
        print("No runnable tasks - did the download scripts run?", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for task in tasks:
            print(f"{task.name}: {len(task.context)} context, {len(task.queries)} queries")
        return

    print(f"Running {len(tasks)} task(s) per model over {len(args.models)} model(s)\n")
    results = run_suite(tasks, args.models)
    sys.exit(0 if report(tasks, args.models, results) else 1)
