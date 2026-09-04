"""Discovers the per-dataset packages under `model_testing/datasets/`.

Each dataset directory is self-contained: a `download.py` that fetches its
images and a `test_<slug>.py` that declares its tasks via `build_tasks()`.
Adding a dataset means adding a directory - nothing here or in
`test_models.py` needs editing.
"""
import importlib.util
from pathlib import Path
from types import ModuleType

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def dataset_dirs() -> list[Path]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(
        d for d in DATASETS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_")) and list(d.glob("test_*.py"))
    )


def _load(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modules() -> list[tuple[str, ModuleType]]:
    """(slug, module) for every dataset that exposes `build_tasks()`."""
    modules = []
    for directory in dataset_dirs():
        script = sorted(directory.glob("test_*.py"))[0]
        module = _load(script, f"pictsure_dataset_{directory.name}")
        if not hasattr(module, "build_tasks"):
            raise AttributeError(f"{script} does not define build_tasks()")
        modules.append((directory.name, module))
    return modules


def download_scripts() -> list[Path]:
    return [d / "download.py" for d in dataset_dirs() if (d / "download.py").exists()]
