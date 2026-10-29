"""PACO-LVIS restricted to tabletop objects (39 objects, 197 object-part classes).

Produced by datasets/paco_tabletop_preprocess.py into $DETECTRON2_DATASETS/PACO_tabletop.
No zero-shot split: every class is trained on, so evaluation reports `mIoU-base` over all classes
(the evaluator skips the `-unbase` metrics when no class is held out).

Registered names:
    paco_tabletop_obj_part_sem_seg_{train,val,test}                 one record per image
    paco_tabletop_obj_part_sem_seg_{train,val,test}_obj_condition   one record per (image, object
        class with part labels); the model is told the object class, as in the paper's protocol
"""
import json
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog

IGNORE_LABEL = 65535
SPLITS = ("train", "val", "test")


def _colors(names, seed):
    rng = random.Random(seed)
    return {n: [rng.randint(0, 255) for _ in range(3)] for n in names}


def _load(root, split, obj_condition):
    records = []
    for r in json.load(open(os.path.join(root, f"{split}.json"))):
        base = {
            "file_name": r["file_name"],
            "height": r["height"],
            "width": r["width"],
            "sem_seg_file_name": r["sem_seg_file_name"],
            # pre-rendered so every part pixel lies inside its own parent object's region
            "obj_sem_seg_file_name": r["obj_sem_seg_file_name"],
        }
        if not obj_condition:
            records.append(base)
            continue
        for obj_class in r["obj_classes_with_parts"]:
            records.append(dict(base, category_id=obj_class))
    return records


def register_paco_tabletop(root):
    root = os.path.join(root, "PACO_tabletop")
    classes_file = os.path.join(root, "classes.json")
    if not os.path.isfile(classes_file):
        return  # dataset not converted on this machine; keep every other dataset usable
    classes = json.load(open(classes_file))
    class_names, obj_names = classes["CLASS_NAMES"], classes["OBJ_CLASS_NAMES"]
    part_names = sorted({c.split("'s")[1].strip() for c in class_names})
    class_colors = _colors(class_names, 0)
    obj_colors = _colors(obj_names, 1)
    part_colors = _colors(part_names, 2)

    meta = {
        "stuff_classes": class_names,
        "obj_classes": obj_names,
        "obj_base_classes": obj_names,
        "obj_eval_classes": obj_names,
        "obj_eval_base_classes": obj_names,
        "part_classes": part_names,
        "part_base_classes": part_names,
        "stuff_colors": [class_colors[c] for c in class_names],
        "obj_colors": [obj_colors[c] for c in obj_names],
        "part_colors": [part_colors[c] for c in part_names],
        "image_root": root,
        "sem_seg_root": root,
        "evaluator_type": "sem_seg",
        "ignore_label": IGNORE_LABEL,
        "evaluation_set": {"base": list(range(len(class_names)))},
        "obj_evaluation_set": {"base": list(range(len(obj_names)))},
        "trainable_flag": [1] * len(class_names),
    }
    for split in SPLITS:
        for obj_condition in (False, True):
            name = f"paco_tabletop_obj_part_sem_seg_{split}" + ("_obj_condition" if obj_condition else "")
            DatasetCatalog.register(name, lambda s=split, oc=obj_condition: _load(root, s, oc))
            MetadataCatalog.get(name).set(**meta)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_paco_tabletop(_root)
