"""The task shapes every dataset in this harness reduces to.

A PictSure evaluation is always "here are K labeled context images per class,
now classify these queries". `FixedTask` is one such evaluation; most datasets
are expressed as a `PooledDataset` that can emit a task for several shot
counts.
"""
from dataclasses import dataclass
from pathlib import Path

# Shot counts a pooled dataset tries by default. A dataset only uses the ones
# that leave at least one held-out query image per class.
DEFAULT_SHOT_CANDIDATES = (1, 3, 5)
DEFAULT_MAX_QUERIES_PER_CLASS = 5


@dataclass
class FixedTask:
    """A single fixed context/query task, e.g. the README's cat/dog example."""

    name: str
    context: list[tuple[Path, int]]
    queries: list[tuple[Path, int]]

    def missing_files(self) -> list[Path]:
        return [p for p, _ in self.context + self.queries if not p.exists()]


@dataclass
class PooledDataset:
    """A dataset with several images per class, usable at multiple shot counts."""

    name: str
    class_images: dict[str, list[Path]]  # class name -> paths (label = index in class_names)
    shot_candidates: tuple[int, ...] = DEFAULT_SHOT_CANDIDATES
    max_queries_per_class: int = DEFAULT_MAX_QUERIES_PER_CLASS

    @property
    def class_names(self) -> list[str]:
        return list(self.class_images.keys())

    def usable_shot_counts(self) -> list[int]:
        if not self.class_images:
            return []
        min_count = min(len(v) for v in self.class_images.values())
        return [k for k in self.shot_candidates if k <= min_count - 1]

    def task_for_shots(self, k: int) -> FixedTask:
        context, queries = [], []
        for label, cls in enumerate(self.class_names):
            imgs = self.class_images[cls]
            context += [(p, label) for p in imgs[:k]]
            queries += [(p, label) for p in imgs[k : k + self.max_queries_per_class]]
        return FixedTask(name=f"{self.name} ({k}-shot)", context=context, queries=queries)

    def tasks(self) -> list[FixedTask]:
        # A single class is not a classification task; this happens when a
        # download was interrupted, and silently scoring 100% would hide it.
        if len(self.class_images) < 2:
            return []
        return [self.task_for_shots(k) for k in self.usable_shot_counts()]


def pooled_from_dir(
    root: Path,
    name: str,
    shot_candidates: tuple[int, ...] = DEFAULT_SHOT_CANDIDATES,
    max_queries_per_class: int = DEFAULT_MAX_QUERIES_PER_CLASS,
    classes: list[str] | None = None,
) -> PooledDataset:
    """Build a PooledDataset from a `root/<class>/<image>` directory tree.

    Class order is taken from `classes` when given, so label indices stay
    stable even if a class directory is missing locally; otherwise the
    directories are used in sorted order.
    """
    if classes is None:
        dirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
    else:
        dirs = [root / c for c in classes]
    class_images = {
        d.name: sorted(p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for d in dirs
        if d.is_dir()
    }
    return PooledDataset(
        name=name,
        class_images={k: v for k, v in class_images.items() if v},
        shot_candidates=shot_candidates,
        max_queries_per_class=max_queries_per_class,
    )
