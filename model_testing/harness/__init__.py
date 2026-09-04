"""Shared plumbing for the PictSure test harness.

The per-dataset scripts under `datasets/` import from here; the two top-level
entry points (`download_datasets.sh`, `test_models.py`) use `registry` to find
those scripts.
"""
from .hf_datasets import download_class_subset, label_names
from .net import download, fetch_bytes, fetch_json, quote, resolve_hf_token
from .runner import MODEL_IDS, cli, markdown_table, report, run_suite, run_task
from .tasks import (
    DEFAULT_MAX_QUERIES_PER_CLASS,
    DEFAULT_SHOT_CANDIDATES,
    FixedTask,
    PooledDataset,
    pooled_from_dir,
)

__all__ = [
    "DEFAULT_MAX_QUERIES_PER_CLASS",
    "DEFAULT_SHOT_CANDIDATES",
    "FixedTask",
    "MODEL_IDS",
    "PooledDataset",
    "cli",
    "download",
    "download_class_subset",
    "fetch_bytes",
    "fetch_json",
    "label_names",
    "markdown_table",
    "pooled_from_dir",
    "quote",
    "report",
    "resolve_hf_token",
    "run_suite",
    "run_task",
]
