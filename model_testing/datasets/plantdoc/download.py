#!/usr/bin/env python3
"""PlantDoc - eight visually similar tomato leaf disease classes.

Pulled from the GitHub repo's `test/` directory listing rather than a fixed
file list, because the filenames are not predictable.

    python3 model_testing/datasets/plantdoc/download.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download, fetch_json, quote  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
CONTENTS_API = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/test"

CLASSES = [
    "Tomato Septoria leaf spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato Early blight leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
]

# Filenames containing %, + or spaces do not survive raw URL resolution, so
# only take the plain ones - there are plenty either way.
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.(jpg|jpeg)$", re.IGNORECASE)


def main() -> None:
    print(f"PlantDoc -> {DATA_DIR} ({len(CLASSES)} classes)")
    for cls in CLASSES:
        entries = fetch_json(f"{CONTENTS_API}/{quote(cls)}")
        for entry in entries:
            if SAFE_NAME_RE.match(entry["name"]):
                download(entry["download_url"], DATA_DIR / cls.replace(" ", "_") / entry["name"])


if __name__ == "__main__":
    main()
