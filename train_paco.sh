#!/bin/bash
# Fine-tune PartCATSeg on PACO-LVIS tabletop (39 objects / 197 object-part classes), then score the
# final checkpoint on PACO's test split. Data: $DETECTRON2_DATASETS/PACO_tabletop, built by
# datasets/paco_tabletop_preprocess.py. Config: configs/zero_shot/partcatseg_paco_tabletop.yaml.
#
#   bash train_paco.sh                                    # full run: 20k iters, batch 4, val every 1k
#   OUT=output/paco_smoke bash train_paco.sh SOLVER.MAX_ITER 20 TEST.EVAL_PERIOD 0   # smoke test
#   NUM_GPUS=2 bash train_paco.sh                         # more GPUs (scale SOLVER.IMS_PER_BATCH too)
#
# Trailing args are detectron2 config overrides (KEY VALUE pairs). Re-running resumes from
# $OUT/last_checkpoint. Refuses to start while another process holds the GPU (shared server);
# FORCE=1 overrides that.
set -euo pipefail
cd "$(dirname "$0")"
source ./pixi_setup.sh

CFG=configs/zero_shot/partcatseg_paco_tabletop.yaml
OUT=${OUT:-output/partcatseg_paco_tabletop}
NUM_GPUS=${NUM_GPUS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation near the 24 GB limit

busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "GPU is in use by other processes; not starting (FORCE=1 to override):"; echo "$busy"; exit 1
fi

run() { pixi run --frozen python train_net.py --num-gpus "$NUM_GPUS" --config-file "$CFG" "$@"; }

# 1. train (evaluates on PACO val every TEST.EVAL_PERIOD iterations and at the end)
run --resume OUTPUT_DIR "$OUT" "$@"

# 2. final checkpoint on PACO test (object-conditioned, batch size 1, as in the paper's protocol)
run --eval-only MODEL.WEIGHTS "$OUT/model_final.pth" OUTPUT_DIR "$OUT/test" \
    DATASETS.TEST "('paco_tabletop_obj_part_sem_seg_test_obj_condition',)"
