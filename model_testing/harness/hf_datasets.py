"""Downloads a per-class image subset of any public Hugging Face image dataset.

Uses the HF dataset viewer API rather than the `datasets` library so the
per-dataset download scripts stay dependency-free and only pull the handful of
images each few-shot task actually needs (a few hundred KB instead of the whole
dataset).

Two strategies, in order:

1. `/filter` with a `where` clause on the label column - one request per class,
   which is what we want. Not every dataset has a filterable index, though.
2. `/rows` paging, bucketing rows by label as they arrive. Slower but works for
   any viewable dataset; used automatically when `/filter` errors out.
"""
from pathlib import Path

from .net import download, fetch_json, quote, resolve_hf_token

API = "https://datasets-server.huggingface.co"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROWS_PAGE = 100
MAX_SCAN_ROWS = 20_000


def _api(endpoint: str, dataset: str, config: str, split: str, extra: str = "") -> str:
    return (
        f"{API}/{endpoint}?dataset={quote(dataset)}&config={quote(config)}"
        f"&split={quote(split)}{extra}"
    )


def label_names(dataset: str, config: str, split: str, label_column: str, token=None) -> list[str]:
    """The ClassLabel names of `label_column`, in label-index order."""
    info = fetch_json(_api("first-rows", dataset, config, split), token=token)
    for feature in info["features"]:
        if feature["name"] == label_column:
            names = feature["type"].get("names")
            if not names:
                raise ValueError(f"{dataset}: column '{label_column}' is not a ClassLabel")
            return list(names)
    raise ValueError(f"{dataset}: no column named '{label_column}'")


def _image_src(cell) -> tuple[str, str]:
    src = cell["src"] if isinstance(cell, dict) else cell
    return src, Path(src.split("?")[0]).suffix.lower() or ".jpg"


def _have(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return len([p for p in class_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES])


def _fetch_by_filter(ds, cfg, split, image_col, label_col, targets, per_class, root, token) -> bool:
    """One /filter request per class. Returns False if the dataset has no index."""
    for label_index, folder in targets.items():
        have = _have(root / folder)
        if have >= per_class:
            continue
        where = quote('"%s"=%d' % (label_col, label_index))
        url = _api(
            "filter", ds, cfg, split,
            f"&where={where}&offset={have}&length={per_class - have}",
        )
        try:
            page = fetch_json(url, token=token)
        except Exception as e:  # noqa: BLE001 - any API failure means: try /rows instead
            print(f"  /filter unavailable ({type(e).__name__}), falling back to a /rows scan")
            return False
        for i, row in enumerate(page["rows"]):
            src, suffix = _image_src(row["row"][image_col])
            # cached-asset URLs are signed and short-lived, so fetch as we page.
            download(src, root / folder / f"{have + i:03d}{suffix}", token=token)
    return True


def _fetch_by_row_scan(ds, cfg, split, image_col, label_col, targets, per_class, root, token):
    """Page through /rows, keeping images only for classes we still need."""
    need = {idx: per_class - _have(root / folder) for idx, folder in targets.items()}
    next_index = {idx: _have(root / folder) for idx, folder in targets.items()}
    offset = 0
    while any(n > 0 for n in need.values()) and offset < MAX_SCAN_ROWS:
        page = fetch_json(
            _api("rows", ds, cfg, split, f"&offset={offset}&length={ROWS_PAGE}"), token=token
        )
        rows = page["rows"]
        if not rows:
            break
        for row in rows:
            label = row["row"][label_col]
            if need.get(label, 0) <= 0:
                continue
            src, suffix = _image_src(row["row"][image_col])
            download(src, root / targets[label] / f"{next_index[label]:03d}{suffix}", token=token)
            next_index[label] += 1
            need[label] -= 1
        offset += len(rows)


def download_class_subset(
    dataset: str,
    dest_root: Path,
    classes: list[str],
    per_class: int,
    *,
    config: str = "default",
    split: str = "test",
    image_column: str = "image",
    label_column: str = "label",
    folder_names: dict[str, str] | None = None,
) -> None:
    """Fetch `per_class` images for each of `classes` into `dest_root/<folder>/`.

    `classes` are ClassLabel *names* from the dataset itself; label indices are
    resolved from the dataset's own feature schema, so a rename upstream fails
    loudly instead of silently mislabeling images.
    """
    token = resolve_hf_token()
    names = label_names(dataset, config, split, label_column, token=token)
    unknown = [c for c in classes if c not in names]
    if unknown:
        raise ValueError(f"{dataset}: unknown class names {unknown}; available: {names}")

    folder_names = folder_names or {}
    targets = {names.index(c): folder_names.get(c, c.replace("/", "_")) for c in classes}

    print(f"{dataset} [{split}] -> {dest_root.name}/ ({len(classes)} classes x {per_class})")
    args = (dataset, config, split, image_column, label_column, targets, per_class,
            dest_root, token)
    if not _fetch_by_filter(*args):
        _fetch_by_row_scan(*args)

    short = {f: _have(dest_root / f) for f in targets.values() if _have(dest_root / f) < per_class}
    if short:
        print(f"  note: fewer images available than requested: {short}")
