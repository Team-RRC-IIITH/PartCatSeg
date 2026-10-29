"""Convert PACO-LVIS into PartCATSeg's object-part format, keeping only tabletop object classes.

    python datasets/paco_tabletop_preprocess.py \
        --paco-ann /scratch2/ennovatex.Soumo/part_datasets/PACO/annotations \
        --coco-root /scratch2/ennovatex.Soumo/part_datasets/coco \
        --out /scratch2/ennovatex.Soumo/part_datasets/PACO_tabletop

Writes, per split (train / val / test, PACO's own split names):
    <out>/annotations_detectron2_part/<split>/<image_id>.png   uint16, pixel = index into CLASS_NAMES,
                                                                65535 = unlabelled
    <out>/annotations_detectron2_obj/<split>/<image_id>.png    uint16, pixel = index into
                                                                OBJ_CLASS_NAMES, 65535 = none
    <out>/<split>.json   one record per image: file_name, height, width, the two map paths, and
                         obj_classes_with_parts
    <out>/classes.json   CLASS_NAMES ("<object>'s <part>") and OBJ_CLASS_NAMES, plus PACO's raw names

Only images containing at least one part of a tabletop object are kept. Objects of other PACO
classes (chair, car, ...) are dropped and therefore count as background, like any other
out-of-vocabulary pixel. Tabletop objects that PACO annotated without parts stay in the object
annotations, so they are not taught as background; their pixels are unlabelled in the part map.
"""
import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# PACO object name -> display name used in the text prompts. Display names must not contain "'s"
# (PartCATSeg splits class names on it), and no object name may be a substring of another, because
# PartCATSeg's object-conditioned inference selects parts with `obj_class in class_name`
# ("pen" would otherwise also match "pencil's ...").
TABLETOP = {
    "ball": "ball",
    "basket": "basket",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "box": "box",
    "calculator": "calculator",
    "can": "can",
    "carton": "carton",
    "cellular_telephone": "cell phone",
    "cup": "cup",
    "drill": "drill",
    "earphone": "earphone",
    "glass_(drink_container)": "drinking glass",
    "hammer": "hammer",
    "jar": "jar",
    "kettle": "kettle",
    "knife": "knife",
    "mouse_(computer_equipment)": "computer mouse",
    "mug": "mug",
    "pan_(for_cooking)": "frying pan",
    "pen": "ink pen",
    "pencil": "pencil",
    "plastic_bag": "plastic bag",
    "plate": "plate",
    "pliers": "pliers",
    "remote_control": "remote control",
    "scissors": "scissors",
    "screwdriver": "screwdriver",
    "soap": "soap",
    "sponge": "sponge",
    "spoon": "spoon",
    "tape_(sticky_cloth_or_paper)": "tape",
    "tissue_paper": "tissue paper",
    "tray": "tray",
    "vase": "vase",
    "wallet": "wallet",
    "watch": "watch",
    "wrench": "wrench",
}
IGNORE = 65535


def part_display(name):
    return name.replace("_", " ")


def check_names(obj_names):
    for a in obj_names:
        assert "'s" not in a, a
        for b in obj_names:
            assert a == b or a not in b, f"object name {a!r} is a substring of {b!r}"


def build_vocab(categories):
    obj_raw = sorted(c["name"] for c in categories if c["supercategory"] == "OBJECT" and c["name"] in TABLETOP)
    obj_names = [TABLETOP[n] for n in obj_raw]
    check_names(obj_names)
    part_raw = sorted(c["name"] for c in categories
                      if c["supercategory"] == "PART" and c["name"].split(":")[0] in TABLETOP)
    class_names = [f"{TABLETOP[p.split(':')[0]]}'s {part_display(p.split(':', 1)[1])}" for p in part_raw]
    assert len(set(class_names)) == len(class_names)
    return obj_raw, obj_names, part_raw, class_names


def rle(seg, h, w):
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):
        return mask_utils.frPyObjects(seg, h, w)
    return seg


def render(job):
    """Draw one image's part map and object map.

    Larger masks are drawn first so small ones (caps, labels, a mug inside a bowl) stay visible.
    Every part pixel then overrides the object map with its own parent object's class. Without
    that, overlapping object masks would leave e.g. mug parts inside the region labelled "bowl",
    and conditioning on "bowl" would supervise/score mug parts.
    """
    part_path, obj_path, h, w, objs, parts = job
    obj = np.full((h, w), IGNORE, dtype=np.uint16)
    for o in sorted(objs, key=lambda o: -o["area"]):
        obj[mask_utils.decode(rle(o["segmentation"], h, w)).astype(bool)] = o["category_id"]
    label = np.full((h, w), IGNORE, dtype=np.uint16)
    for cls, parent, seg, area in sorted(parts, key=lambda p: -p[3]):
        m = mask_utils.decode(rle(seg, h, w)).astype(bool)
        label[m] = cls
        obj[m] = parent
    Image.fromarray(label).save(part_path)
    Image.fromarray(obj).save(obj_path)
    return int((label != IGNORE).sum())


def convert_split(split, args, vocab):
    obj_raw, obj_names, part_raw, class_names = vocab
    d = json.load(open(os.path.join(args.paco_ann, f"paco_lvis_v1_{split}.json")))
    cats = {c["id"]: c for c in d["categories"]}
    obj_idx = {n: i for i, n in enumerate(obj_raw)}
    part_idx = {n: i for i, n in enumerate(part_raw)}

    objs, parts, obj_with_parts = defaultdict(list), defaultdict(list), defaultdict(set)
    for a in d["annotations"]:
        c = cats[a["category_id"]]
        if c["name"].split(":")[0] not in TABLETOP:
            continue
        if c["supercategory"] == "OBJECT":
            objs[a["image_id"]].append({"id": a["id"], "category_id": obj_idx[c["name"]], "bbox": a["bbox"],
                                        "segmentation": a["segmentation"], "area": a["area"]})
        else:
            parent = obj_idx[c["name"].split(":")[0]]
            parts[a["image_id"]].append((part_idx[c["name"]], parent, a["segmentation"], a["area"]))
            obj_with_parts[a["image_id"]].add(parent)

    part_dir = os.path.join(args.out, "annotations_detectron2_part", split)
    obj_dir = os.path.join(args.out, "annotations_detectron2_obj", split)
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)
    records, jobs = [], []
    for im in d["images"]:
        if not parts.get(im["id"]):
            continue
        part_png = os.path.join(part_dir, f"{im['id']}.png")
        obj_png = os.path.join(obj_dir, f"{im['id']}.png")
        records.append({
            "image_id": im["id"],
            "file_name": os.path.join(args.coco_root, im["file_name"]),
            "height": im["height"],
            "width": im["width"],
            "sem_seg_file_name": part_png,
            "obj_sem_seg_file_name": obj_png,
            # object classes with at least one part mask here: the object-conditioned eval
            # needs part ground truth for the class it conditions on
            "obj_classes_with_parts": sorted(obj_with_parts[im["id"]]),
        })
        jobs.append((part_png, obj_png, im["height"], im["width"], objs[im["id"]], parts[im["id"]]))

    with Pool(args.workers) as pool:
        labelled = pool.map(render, jobs, chunksize=64)
    with open(os.path.join(args.out, f"{split}.json"), "w") as f:
        json.dump(records, f)
    present = sorted({p[0] for job in jobs for p in job[5]})
    missing = [r["file_name"] for r in records if not os.path.isfile(r["file_name"])]
    print(f"{split}: {len(records)} images, {sum(len(j[4]) for j in jobs)} objects, "
          f"{sum(len(j[5]) for j in jobs)} part masks, {len(present)}/{len(class_names)} part classes present, "
          f"{np.mean(labelled):.0f} labelled px/image, {len(missing)} images not downloaded yet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paco-ann", required=True)
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    categories = json.load(open(os.path.join(args.paco_ann, "paco_lvis_v1_val.json")))["categories"]
    vocab = build_vocab(categories)
    obj_raw, obj_names, part_raw, class_names = vocab
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump({"CLASS_NAMES": class_names, "OBJ_CLASS_NAMES": obj_names,
                   "paco_part_names": part_raw, "paco_obj_names": obj_raw}, f, indent=1)
    print(f"vocabulary: {len(obj_names)} objects, {len(class_names)} object-part classes")
    for split in args.splits:
        convert_split(split, args, vocab)


if __name__ == "__main__":
    main()
