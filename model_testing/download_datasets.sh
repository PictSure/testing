#!/usr/bin/env bash
# Downloads the few-shot image classification example datasets used by
# test_models.py from the PictSure/pictsure-library GitHub repo.
#
# Datasets are NOT committed to this repo (see .gitignore) - run this
# script whenever you need them locally:
#
#   ./model_testing/download_datasets.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/datasets"
BASE_URL="https://raw.githubusercontent.com/PictSure/pictsure-library/main/Examples"

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "skip (exists): $dest"
    return
  fi
  echo "fetching: $dest"
  curl -sf "$url" -o "$dest"
}

# --- CatsDogs: binary cat/dog classification ---
mkdir -p "$DATASET_DIR/CatsDogs"
for f in cat1.jpg cat2.jpg dog1.jpg dog2.jpg query.jpg; do
  fetch "$BASE_URL/CatsDogs/$f" "$DATASET_DIR/CatsDogs/$f"
done

# --- BrainTumor_preprocessed: 4-way MRI tumor classification ---
for cls in glioma meningioma notumor pituitary; do
  mkdir -p "$DATASET_DIR/BrainTumor_preprocessed/$cls"
  prefix="${cls:0:2}"
  # Prefixes don't all match the first two letters (notumor -> "no", etc.)
  case "$cls" in
    glioma) prefix="gl" ;;
    meningioma) prefix="me" ;;
    notumor) prefix="no" ;;
    pituitary) prefix="pi" ;;
  esac
  for i in $(seq -w 10 29); do
    f="Te-${prefix}_00${i}.jpg"
    fetch "$BASE_URL/BrainTumor_preprocessed/$cls/$f" "$DATASET_DIR/BrainTumor_preprocessed/$cls/$f"
  done
done

# --- PlantDoc (tomato leaf disease subset) and SwedishFlowers (wildflower subset) ---
python3 "$SCRIPT_DIR/download_extra_datasets.py"

echo "Done. Datasets are in $DATASET_DIR"
