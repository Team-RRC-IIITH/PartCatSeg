#!/bin/bash
# Download PACO-LVIS annotations plus only the COCO 2017 images PACO uses (~57.6k images, ~9 GB).
#
#   bash datasets/paco_download.sh [ROOT]      # ROOT default: $DETECTRON2_DATASETS, else ./datasets
#
# Writes ROOT/PACO/annotations/paco_lvis_v1_{train,val,test}.json and ROOT/coco/{train,val}2017/.
# Images are fetched one per connection, 48 in parallel: a single connection to
# images.cocodataset.org can be very slow (~67 KB/s measured), which makes the full 20 GB COCO zips
# a multi-day download. Resumable: re-run to retry failures; finished files are skipped.
set -euo pipefail
ROOT=$(realpath "${1:-${DETECTRON2_DATASETS:-datasets}}")
ANN=$ROOT/PACO/annotations
mkdir -p "$ANN" "$ROOT/coco/train2017" "$ROOT/coco/val2017"

cd "$ANN"
wget -c -q https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip
echo "02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80  paco_lvis_v1.zip" | sha256sum -c -
[ -e .unzipped ] || { unzip -q -o paco_lvis_v1.zip && touch .unzipped; }

cd "$ROOT/coco"
LIST=$ROOT/PACO/image_list.txt
python3 - "$ANN" > "$LIST" <<'PY'
import json, sys
seen = set()
for split in ("train", "val", "test"):
    for im in json.load(open(f"{sys.argv[1]}/paco_lvis_v1_{split}.json"))["images"]:
        if im["file_name"] not in seen:
            seen.add(im["file_name"]); print(im["coco_url"], im["file_name"])
PY
TOTAL=$(wc -l < "$LIST")
echo "$(date) fetching up to $TOTAL images into $ROOT/coco"
awk '{ if (system("test -s " $2) != 0) print }' "$LIST" \
  | xargs -P 48 -n 2 sh -c 'curl -sf --retry 5 --retry-delay 2 -o "$1.part" "$0" && mv "$1.part" "$1" || echo "FAILED $0"'
HAVE=$(awk '{ if (system("test -s " $2) == 0) n++ } END { print n+0 }' "$LIST")
echo "$(date) $HAVE / $TOTAL images present"
[ "$HAVE" -eq "$TOTAL" ] && echo "ALL DONE" || { echo "INCOMPLETE - re-run to retry"; exit 1; }
