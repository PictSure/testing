"""
Downloads two additional few-shot classification datasets for test_models.py:

- PlantDoc (a subset of visually-similar tomato leaf disease classes, from
  https://github.com/pratikkayal/PlantDoc-Dataset)
- SwedishFlowers (a subset of wildflower species, from the HF dataset
  renukadevichappidi/swedish-flowers-dataset)

Called by download_datasets.sh - not meant to be run standalone against
arbitrary datasets.
"""
import re
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "datasets"
HF_TOKEN = Path("/home/cornelius/.hf_credentials").read_text().strip()

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.(jpg|jpeg|JPG|JPEG)$")

PLANTDOC_CLASSES = [
    "Tomato Septoria leaf spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato Early blight leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
]

SWEDISH_FLOWER_CLASSES = [
    "Prästkrage",  # oxeye daisy
    "Vitsippa",  # wood anemone
    "Ljung",  # heather
    "Smörblomma",  # buttercup
    "Liljekonvalj",  # lily of the valley
]


def fetch_json(url: str, token: str | None = None) -> object:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    import json

    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def download(url: str, dest: Path, token: str | None = None) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"fetched: {dest}")


def download_plantdoc():
    base_api = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/contents/test"
    for cls in PLANTDOC_CLASSES:
        local_cls = cls.replace(" ", "_")
        entries = fetch_json(f"{base_api}/{urllib.parse.quote(cls)}")
        for entry in entries:
            name = entry["name"]
            if not SAFE_NAME_RE.match(name):
                continue  # skip filenames with %, +, spaces etc. that break raw URL resolution
            dest = DATASET_DIR / "PlantDoc" / local_cls / name
            download(entry["download_url"], dest)


def download_swedish_flowers():
    repo = "renukadevichappidi/swedish-flowers-dataset"
    for cls in SWEDISH_FLOWER_CLASSES:
        quoted_cls = urllib.parse.quote(cls)
        entries = fetch_json(
            f"https://huggingface.co/api/datasets/{repo}/tree/main/data/test/{quoted_cls}",
            token=HF_TOKEN,
        )
        for entry in entries:
            name = entry["path"].split("/")[-1]
            if not re.match(r"^\d+\.jpg$", name):
                continue  # skip "aug_*" augmented duplicates, keep only original photos
            dest = DATASET_DIR / "SwedishFlowers" / cls / name
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/test/{quoted_cls}/{name}"
            download(url, dest, token=HF_TOKEN)


if __name__ == "__main__":
    download_plantdoc()
    download_swedish_flowers()
    print("Done.")
