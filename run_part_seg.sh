#!/bin/bash
# Per-object part segmentation of one image with PartCATSeg, using
#   gt   : GraspClutter6D ground-truth masks (mask_visib/<frame>_*.png)
#   sam3 : SAM 3 masks, detected first with export_sam3_masks.py
#   both : gt, then sam3
# Edit the paths below, then:  bash run_part_seg.sh
# Results: results/<scene>/<frame>_paco_{mask_visib,sam3_masks}_{parts,objects,labels}.png + _parts.json
set -euo pipefail

# ------------------------------------------------------------------ edit here
MODE=both                                                        # gt | sam3 | both
SCENE_DIR=/scratch2/ennovatex.Rath/dataset/GraspClutter6D-2/scenes/000003
FRAME=000001
IMAGE=$SCENE_DIR/rgb/$FRAME.png                                  # any image works for sam3
GT_MASKS="$SCENE_DIR/mask_visib/${FRAME}_*.png"                  # gt only

WEIGHTS=output/partcatseg_paco_tabletop/model_final.pth          # PartCATSeg checkpoint
SAM3_PROMPT=item                                                 # concrete singular noun ("objects" finds nothing)
EXTRA_ARGS=()                                                    # e.g. (--min-score 0.1 --crop-margin 0.25)

SOM_PYTHON=/scratch2/ennovatex.Soumo/SegObjectMaterial/.pixi/envs/default/bin/python   # has SAM 3
HF_CACHE=/scratch2/ennovatex.Soumo/.cache/huggingface            # holds facebook/sam3
# ----------------------------------------------------------------------------

cd "$(dirname "$0")"
source ./pixi_setup.sh
[ -f "$IMAGE" ] || { echo "image not found: $IMAGE"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "checkpoint not found: $WEIGHTS"; exit 1; }

# same rule as infer_image.py: scenes/<id>/rgb/<frame>.png -> <id>, else the image's folder name
stem=$(basename "${IMAGE%.*}")
parent=$(basename "$(dirname "$IMAGE")")
if [ "$parent" = rgb ]; then scene=$(basename "$(dirname "$(dirname "$IMAGE")")"); else scene=$parent; fi

partseg() {  # partseg <mask png>...
    pixi run --frozen python infer_image.py "$IMAGE" --models paco --weights "$WEIGHTS" \
        --obj-masks "$@" "${EXTRA_ARGS[@]}"
}

if [ "$MODE" = gt ] || [ "$MODE" = both ]; then
    # shellcheck disable=SC2206  # the glob must expand here
    masks=($GT_MASKS)
    [ -e "${masks[0]}" ] || { echo "no GT masks match: $GT_MASKS"; exit 1; }
    echo "=== GT masks: ${#masks[@]}"
    partseg "${masks[@]}"
fi

if [ "$MODE" = sam3 ] || [ "$MODE" = both ]; then
    out=results/$scene/sam3_masks
    echo "=== SAM 3 masks ('$SAM3_PROMPT') -> $out"
    # SegObjectMaterial's env, called directly: its pixi.toml points HF_HOME at a stale path
    HF_HOME=$HF_CACHE HF_HUB_OFFLINE=1 "$SOM_PYTHON" export_sam3_masks.py "$IMAGE" \
        --prompt "$SAM3_PROMPT" --out "$out"
    masks=("$out/${stem}"_sam3_*.png)
    [ -e "${masks[0]}" ] || { echo "SAM 3 found no objects; try another SAM3_PROMPT"; exit 1; }
    partseg "${masks[@]}"
fi

echo "=== done: results/$scene/"
