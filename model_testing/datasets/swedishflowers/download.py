#!/usr/bin/env python3
"""SwedishFlowers - five wildflower species, visually distinct.

Fetched from the Hugging Face repo file tree (not the dataset viewer) because
the images sit in plain `data/test/<class>/` folders.

    python3 model_testing/datasets/swedishflowers/download.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import download, fetch_json, quote, resolve_hf_token  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
REPO = "renukadevichappidi/swedish-flowers-dataset"

# ASCII folder name -> the Swedish name as it appears upstream, with the
# English gloss for readers.
CLASS_NAMES = {
    "Prastkrage": "Prästkrage",  # oxeye daisy
    "Vitsippa": "Vitsippa",  # wood anemone
    "Ljung": "Ljung",  # heather
    "Smorblomma": "Smörblomma",  # buttercup
    "Liljekonvalj": "Liljekonvalj",  # lily of the valley
}

# The upstream folders also hold "aug_*" augmented copies; keep originals only.
ORIGINAL_RE = re.compile(r"^\d+\.jpg$")


def main() -> None:
    token = resolve_hf_token()
    print(f"SwedishFlowers -> {DATA_DIR} ({len(CLASS_NAMES)} classes)")
    for folder, upstream in CLASS_NAMES.items():
        quoted = quote(upstream)
        entries = fetch_json(
            f"https://huggingface.co/api/datasets/{REPO}/tree/main/data/test/{quoted}",
            token=token,
        )
        for entry in entries:
            name = entry["path"].split("/")[-1]
            if not ORIGINAL_RE.match(name):
                continue
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/test/{quoted}/{name}"
            download(url, DATA_DIR / folder / name, token=token)


if __name__ == "__main__":
    main()
