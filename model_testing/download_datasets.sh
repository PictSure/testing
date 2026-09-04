#!/usr/bin/env bash
# Downloads the images for every dataset in this harness by running each
# dataset's own download script under model_testing/datasets/<name>/.
#
# Images are NOT committed to this repo (see .gitignore) - run this whenever
# you need them locally:
#
#   ./model_testing/download_datasets.sh                 # every dataset
#   ./model_testing/download_datasets.sh beans dtd       # just these
#
# To fetch a single dataset directly, run its own script instead:
#
#   python3 model_testing/datasets/beans/download.py
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"
PYTHON="${PYTHON:-python3}"

if [ "$#" -gt 0 ]; then
  scripts=()
  for name in "$@"; do
    script="$DATASETS_DIR/$name/download.py"
    if [ ! -f "$script" ]; then
      echo "error: no such dataset '$name' (expected $script)" >&2
      exit 2
    fi
    scripts+=("$script")
  done
else
  # Sorted so runs are reproducible; a dataset is just a directory with a
  # download.py, so adding one needs no change here.
  IFS=$'\n' read -r -d '' -a scripts < <(find "$DATASETS_DIR" -mindepth 2 -maxdepth 2 -name download.py | sort && printf '\0')
fi

failed=()
for script in "${scripts[@]}"; do
  name="$(basename "$(dirname "$script")")"
  echo "=== $name ==="
  if ! "$PYTHON" "$script"; then
    echo "!!! $name failed" >&2
    failed+=("$name")
  fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "Done with errors. Failed dataset(s): ${failed[*]}" >&2
  echo "Datasets that did download are usable; re-run to retry the rest."
  exit 1
fi
echo "Done. ${#scripts[@]} dataset(s) downloaded under $DATASETS_DIR/<name>/data/"
