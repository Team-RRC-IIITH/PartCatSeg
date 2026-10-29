#!/usr/bin/env python
"""Detect objects with SAM 3 and write one binary PNG per object, for infer_image.py --obj-masks.

SAM 3 needs transformers >= 5, which this repo's env doesn't have, so run it with
SegObjectMaterial's env (the weights are already cached under /scratch2/ennovatex.Soumo/.cache):

    SOM=/scratch2/ennovatex.Soumo/SegObjectMaterial
    HF_HOME=/scratch2/ennovatex.Soumo/.cache/huggingface HF_HUB_OFFLINE=1 \
        $SOM/.pixi/envs/default/bin/python export_sam3_masks.py <image> [--out DIR]

Writes <out>/<stem>_sam3_<k>.png (255 = object), default out = results/<scene>/sam3_masks/.
Detection settings follow SegObjectMaterial's detect_objects_sam3 (utils/part_common.py): prompt
"item" (abstract plurals such as "objects" score just under the threshold and return nothing),
threshold 0.5, masks under 2000 px dropped, and masks over 15 % of the image dropped (on GC6D that
is the tote, not an object).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
# this repo vendors an old, partial transformers/ that would shadow the env's transformers 5.x
sys.path = [p for p in sys.path if Path(p or ".").resolve() != REPO]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def scene_name_for(image_path):
    """GraspClutter6D-style scenes/<id>/rgb/<frame>.png -> <id>; otherwise the parent dir name."""
    return image_path.parent.parent.name if image_path.parent.name == "rgb" else image_path.parent.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", help="output dir (default: results/<scene>/sam3_masks)")
    ap.add_argument("--prompt", default="item", help="SAM 3 concept; use a concrete singular noun")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=2000, help="drop masks smaller than this (px)")
    ap.add_argument("--max-area-frac", type=float, default=0.15, help="drop masks larger than this")
    args = ap.parse_args()

    from transformers import Sam3Model, Sam3Processor

    image_path = Path(args.image).resolve()
    out = Path(args.out) if args.out else REPO / "results" / scene_name_for(image_path) / "sam3_masks"
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=args.threshold, mask_threshold=0.5, target_sizes=[(h, w)])[0]

    # stale masks from an earlier run would otherwise be picked up by a *_sam3_*.png glob
    for old in out.glob(f"{image_path.stem}_sam3_*.png"):
        old.unlink()
    kept, small, big = 0, 0, 0
    for score, mask in sorted(zip(res["scores"].tolist(), res["masks"]), key=lambda t: -t[0]):
        mask = mask.cpu().numpy().astype(bool)
        if mask.sum() < args.min_area:
            small += 1
            continue
        if mask.sum() > args.max_area_frac * w * h:
            big += 1
            continue
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{image_path.stem}_sam3_{kept:03d}.png")
        print(f"  {image_path.stem}_sam3_{kept:03d}.png  score {score:.3f}  {int(mask.sum())} px")
        kept += 1
    print(f"SAM 3 '{args.prompt}': {len(res['scores'])} masks -> {kept} objects "
          f"({big} too large, {small} too small) in {out}")


if __name__ == "__main__":
    main()
