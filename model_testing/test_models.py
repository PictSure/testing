"""
Sanity-checks that every PictSure model repo actually works with the few-shot
in-context classification usage pattern shown in its README, across multiple
datasets and shot counts, before README updates get pushed to the Hub.

Datasets are not committed to the repo - run download_datasets.sh first:

    ./model_testing/download_datasets.sh
    python3 model_testing/test_models.py

The models are public, so no Hugging Face token is required, but providing
one avoids anonymous rate limits. Set it via the HF_TOKEN environment
variable, or run `huggingface-cli login` beforehand.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PictSure import PictSure


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return cached.read_text().strip()
    return None


TOKEN = resolve_hf_token()

MODEL_IDS = [
    "pictsure/pictsure-vit",
    "pictsure/pictsure-resnet",
    "pictsure/pictsure-dinov2",
    "pictsure/pictsure-dinov2-large",
    "pictsure/pictsure-clip",
]

DATASET_DIR = Path(__file__).parent / "datasets"

# Shot counts to try for pooled datasets; a dataset only uses the ones that
# leave at least one held-out query image per class.
SHOT_CANDIDATES = [1, 3, 5, 10]
MAX_QUERIES_PER_CLASS = 5


@dataclass
class FixedTask:
    """A single fixed context/query task, e.g. the README's cat/dog example."""

    name: str
    context: list[tuple[Path, int]]
    queries: list[tuple[Path, int]]


@dataclass
class PooledDataset:
    """A dataset with several images per class, usable at multiple shot counts."""

    name: str
    class_images: dict[str, list[Path]]  # class_name -> image paths (label = index in class_names)

    @property
    def class_names(self) -> list[str]:
        return list(self.class_images.keys())

    def usable_shot_counts(self) -> list[int]:
        min_count = min(len(v) for v in self.class_images.values())
        return [k for k in SHOT_CANDIDATES if k <= min_count - 1]

    def task_for_shots(self, k: int) -> FixedTask:
        context, queries = [], []
        for label, cls in enumerate(self.class_names):
            imgs = self.class_images[cls]
            context += [(p, label) for p in imgs[:k]]
            queries += [(p, label) for p in imgs[k : k + MAX_QUERIES_PER_CLASS]]
        return FixedTask(name=f"{self.name} ({k}-shot)", context=context, queries=queries)


def cats_dogs_task() -> FixedTask:
    d = DATASET_DIR / "CatsDogs"
    return FixedTask(
        name="CatsDogs (README example, 2-shot)",
        context=[(d / "cat1.jpg", 0), (d / "cat2.jpg", 0), (d / "dog1.jpg", 1), (d / "dog2.jpg", 1)],
        queries=[(d / "query.jpg", 1)],
    )


def brain_tumor_dataset() -> PooledDataset:
    d = DATASET_DIR / "BrainTumor_preprocessed"
    prefixes = {"glioma": "gl", "meningioma": "me", "notumor": "no", "pituitary": "pi"}
    class_images = {
        cls: [d / cls / f"Te-{prefix}_00{i}.jpg" for i in range(10, 30)]
        for cls, prefix in prefixes.items()
    }
    return PooledDataset(name="BrainTumor_preprocessed", class_images=class_images)


def plantdoc_dataset() -> PooledDataset:
    d = DATASET_DIR / "PlantDoc"
    class_images = {p.name: sorted(p.glob("*")) for p in sorted(d.iterdir()) if p.is_dir()}
    return PooledDataset(name="PlantDoc", class_images=class_images)


def swedish_flowers_dataset() -> PooledDataset:
    d = DATASET_DIR / "SwedishFlowers"
    class_images = {p.name: sorted(p.glob("*")) for p in sorted(d.iterdir()) if p.is_dir()}
    return PooledDataset(name="SwedishFlowers", class_images=class_images)


def build_tasks() -> list[FixedTask]:
    tasks = [cats_dogs_task()]
    for dataset in [brain_tumor_dataset(), plantdoc_dataset(), swedish_flowers_dataset()]:
        for k in dataset.usable_shot_counts():
            tasks.append(dataset.task_for_shots(k))
    return tasks


def run_task(model: PictSure, task: FixedTask) -> tuple[bool, str]:
    try:
        context_images = [Image.open(p) for p, _ in task.context]
        context_labels = [label for _, label in task.context]
        model.set_context_images(context_images, context_labels)

        correct = 0
        for path, expected in task.queries:
            prediction = model.predict(Image.open(path))
            correct += int(prediction == expected)

        total = len(task.queries)
        return True, f"{correct}/{total} correct"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset dir {DATASET_DIR} missing - run download_datasets.sh first.")
        sys.exit(1)

    tasks = build_tasks()
    print(f"Running {len(tasks)} tasks per model: {[t.name for t in tasks]}\n")

    results = {}
    for model_id in MODEL_IDS:
        print(f"=== {model_id} ===", flush=True)
        try:
            model = PictSure.from_pretrained(model_id, token=TOKEN)
        except Exception as e:  # noqa: BLE001
            for task in tasks:
                results[(model_id, task.name)] = (False, f"load failed: {type(e).__name__}: {e}")
            continue

        for task in tasks:
            ok, msg = run_task(model, task)
            results[(model_id, task.name)] = (ok, msg)
            print(f"{'OK  ' if ok else 'FAIL'} {task.name}: {msg}", flush=True)
        print(flush=True)

    print("=== Summary ===")
    all_ok = True
    for (model_id, task_name), (ok, msg) in results.items():
        print(f"{'OK  ' if ok else 'FAIL'} {model_id} / {task_name}: {msg}")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
