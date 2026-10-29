#!/usr/bin/env python
"""Run PartCATSeg on one image and save part-segmentation visualisations with part names drawn on.

    pixi run python infer_image.py /path/to/img.png                                   # PACO model
    pixi run python infer_image.py <img> --weights output/.../model_0009999.pth       # any checkpoint
    pixi run python infer_image.py <img> --vocab "mug's handle" "mug's body" "lid's knob"   # zero-shot
    pixi run python infer_image.py <img> --vocab-file my_parts.txt                    # one per line
    pixi run python infer_image.py /scratch2/GraspClutter6D-2/scenes/000000 --models voc ade
    pixi run python infer_image.py <img> --objects mug bottle                         # restrict vocab
    pixi run python infer_image.py <img> --min-score 0.05                             # drop weak labels
    pixi run python infer_image.py $S/rgb/000001.png --obj-masks $S/mask_visib/000001_*.png   # per object

Results go to results/<scene-name>/ (scene name = the dataset scene directory, e.g. 000000).

--obj-masks runs the model once per object: a square crop around each mask, resized to 384^2 (the
training crop size), conditioned on the object class the model scores highest inside the mask, as
the benchmark conditions on the ground-truth class. Only pixels inside a mask get a part label.

Unlike the benchmark evaluation, which is conditioned on the ground-truth object class, this runs
over the model's whole part vocabulary, since a custom image has no ground truth. `--objects`
restores that conditioning by hand: only parts of the listed objects can be predicted.

--vocab / --vocab-file replace the model's training vocabulary with any "<object>'s <part>" names.
The text embeddings are computed from those names when the model is built, so nothing is retrained:
this is the open-vocabulary (zero-shot) setting, and names never seen in training are allowed.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.modeling.postprocessing import sem_seg_postprocess  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.structures import ImageList  # noqa: E402
from detectron2.utils.visualizer import ColorMode, Visualizer  # noqa: E402

from baselines import add_mask_former_config  # noqa: E402,F401  (also registers model + datasets)

# name -> (config, default checkpoint), both relative to the repo root
MODELS = {
    "voc": ("configs/zero_shot/partcatseg_voc.yaml", "weights/partcatseg_voc.pth"),
    "ade": ("configs/zero_shot/partcatseg_ade.yaml", "weights/partcatseg_ade.pth"),
    "paco": ("configs/zero_shot/partcatseg_paco_tabletop.yaml",
             "output/partcatseg_paco_tabletop/model_final.pth"),
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
BACKGROUND = np.iinfo(np.uint16).max  # label value for "no part"


def resolve_image(path):
    """A file is used as-is; a directory yields its first image (looking inside rgb/ if present)."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        sys.exit(f"not found: {path}")
    search = path / "rgb" if (path / "rgb").is_dir() else path
    images = sorted(p for p in search.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"no images in {search}")
    return images[0]


def scene_name_for(image_path):
    """GraspClutter6D-style scenes/<id>/rgb/<frame>.png -> <id>; otherwise the parent dir name."""
    parent = image_path.parent
    if parent.name in {"rgb", "images", "image", "color"}:
        parent = parent.parent
    return parent.name or image_path.stem


def read_vocab(entries, vocab_file):
    """--vocab entries plus --vocab-file lines (blank lines and # comments skipped), normalised to
    "<object>'s <part>", the form PartCATSeg splits into object and part names."""
    lines = list(entries or [])
    if vocab_file:
        lines += [l.split("#")[0] for l in Path(vocab_file).read_text().splitlines()]
    vocab = []
    for line in filter(None, (l.strip() for l in lines)):
        obj, sep, part = line.partition("'s")
        if not sep or not obj.strip() or not part.strip() or "'s" in part:
            sys.exit(f"vocab entry must look like \"<object>'s <part>\" (e.g. \"mug's handle\"): {line!r}")
        name = f"{obj.strip()}'s {part.strip()}"
        if name not in vocab:
            vocab.append(name)
    return vocab


def register_vocab(vocab):
    """Metadata-only entry that PartCATSeg's init_metadata reads the test vocabulary from."""
    name = "custom_vocab_" + "|".join(vocab)
    meta = MetadataCatalog.get(name)
    if not hasattr(meta, "stuff_classes"):
        meta.set(stuff_classes=vocab, obj_classes=sorted({c.split("'s")[0] for c in vocab}),
                 ignore_label=int(BACKGROUND))
    return name


def load_model(name, ckpt=None, test_dataset=None):
    config, default_ckpt = MODELS[name]
    ckpt = Path(ckpt or REPO / default_ckpt)
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(REPO / config))
    # DINOv2Backbone also reads MODEL.WEIGHTS for its own tensors, so point it at the checkpoint
    # rather than at pretrain_weights/model_final_base.pth (only needed for training).
    cfg.MODEL.WEIGHTS = str(ckpt)
    if test_dataset:  # the text embeddings of the test vocabulary are built from this dataset's names
        cfg.DATASETS.TEST = (test_dataset,)
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    return cfg, model


@torch.no_grad()
def forward(cfg, model, image_rgb, size=None):
    """Sigmoid scores at the image's resolution, on the GPU: (specific parts + background,
    generic parts + objects + background). Mirrors PartCATSeg.forward's inference branch.
    The image gets the test-time resize, or is resized to size x size when `size` is given."""
    h, w = image_rgb.shape[:2]
    if size:
        resized = T.ResizeTransform(h, w, size, size).apply_image(image_rgb)
    else:
        resize = T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
        resized = resize.get_transform(image_rgb).apply_image(image_rgb)
    x = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().to(model.device)

    clip_images = ImageList.from_tensors([(x - model.clip_pixel_mean) / model.clip_pixel_std],
                                         model.size_divisibility)
    images = ImageList.from_tensors([(x - model.pixel_mean) / model.pixel_std], model.size_divisibility)
    clip_input = F.interpolate(clip_images.tensor, size=model.clip_resolution, mode="bilinear",
                               align_corners=False)
    clip_features = model.sem_seg_head.predictor.clip_model.encode_image(clip_input, dense=True)
    features = model.backbone(images)
    part_obj, specific, _ = model.sem_seg_head(clip_features, features)

    # The logits cover the padded input (ImageList pads to a multiple of size_divisibility) at a
    # lower resolution. Drop the padded rows/columns before upsampling; sem_seg_postprocess can't,
    # because it crops with input-pixel sizes on the low-resolution grid, i.e. not at all.
    (ih, iw), (ph, pw) = images.image_sizes[0], images.tensor.shape[-2:]

    def upsample(logits):
        oh, ow = logits.shape[-2:]
        logits = logits[..., :round(oh * ih / ph), :round(ow * iw / pw)]
        return F.interpolate(logits.sigmoid(), size=(h, w), mode="bilinear", align_corners=False)[0]

    return upsample(specific), upsample(part_obj)


def predict(cfg, model, image_rgb):
    """Per-pixel sigmoid scores over the full part vocabulary (+ background channel), at the
    original image resolution."""
    return forward(cfg, model, image_rgb)[0].cpu().numpy()


def load_masks(paths, shape):
    """Binary object masks (any nonzero pixel = object), e.g. GraspClutter6D mask_visib PNGs."""
    masks = []
    for p in map(Path, paths):
        m = np.asarray(Image.open(p))
        m = (m if m.ndim == 2 else m[..., 0]) > 0
        if m.shape != shape:
            sys.exit(f"mask {p} is {m.shape[1]}x{m.shape[0]}, image is {shape[1]}x{shape[0]}")
        masks.append((p, m))
    return masks


def gc6d_obj_id(mask_path):
    """GraspClutter6D obj_id for mask_visib/<frame>_<inst>.png, from the scene's scene_gt.json."""
    gt_file = mask_path.parent.parent / "scene_gt.json"
    try:
        frame, inst = (int(s) for s in mask_path.stem.split("_"))
        return json.loads(gt_file.read_text())[str(frame)][inst]["obj_id"]
    except (ValueError, OSError, KeyError, IndexError):
        return None


def crop_box(mask, margin):
    """Square box around the mask, padded by `margin` x its size on each side, kept inside the
    image (it becomes a rectangle only when the object is bigger than the image's short side)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(round(max(y1 - y0, x1 - x0) * (1 + 2 * margin)))
    sy, sx = min(side, h), min(side, w)
    top = int(np.clip((y0 + y1 - sy) // 2, 0, h - sy))
    left = int(np.clip((x0 + x1 - sx) // 2, 0, w - sx))
    return top, top + sy, left, left + sx


def label_map_per_object(cfg, model, image, masks, class_names, objects, min_score, crop_size, margin,
                         min_mask_area):
    """Object-conditioned inference, as in the benchmark but with the object class predicted.
    For each mask: crop around it, resize the crop to crop_size^2, take the object class with the
    highest mean object score inside the mask, then label the mask's pixels with that object's
    parts only. Pixels outside every mask stay BACKGROUND. Larger masks are painted first."""
    predictor = model.sem_seg_head.predictor
    n = len(class_names)
    n_part = len(predictor.test_part_classes)
    obj_names = list(predictor.test_obj_in_part_classes)
    owner = np.array([c.split("'s")[0].strip() for c in class_names])
    candidates = [i for i, o in enumerate(obj_names) if not objects or o in objects]

    labels = np.full(image.shape[:2], BACKGROUND, dtype=np.uint16)
    best = np.zeros(image.shape[:2], dtype=np.float32)
    records = []
    for path, mask in sorted(masks, key=lambda pm: -pm[1].sum()):
        record = {"mask": str(path), "area_px": int(mask.sum()), "gc6d_obj_id": gc6d_obj_id(path)}
        records.append(record)
        if mask.sum() < min_mask_area:
            record["skipped"] = f"smaller than {min_mask_area} px"
            continue
        t, b, l, r = crop_box(mask, margin)
        specific, part_obj = forward(cfg, model, image[t:b, l:r], crop_size)
        m = torch.as_tensor(mask[t:b, l:r], device=specific.device)

        obj_scores = part_obj[n_part:n_part + len(obj_names)][:, m].mean(1).cpu().numpy()
        ranked = sorted(candidates, key=lambda i: -obj_scores[i])
        obj = obj_names[ranked[0]]

        part_scores = specific[:n][:, m]                       # (n, mask pixels)
        part_scores[torch.as_tensor(owner != obj, device=m.device)] = -1.0
        score, label = (v.cpu().numpy() for v in part_scores.max(0))
        label = label.astype(np.uint16)
        label[score < min_score] = BACKGROUND
        labels[t:b, l:r][mask[t:b, l:r]] = label
        best[t:b, l:r][mask[t:b, l:r]] = score

        record.update({
            "crop_tlbr": [t, l, b, r],
            "object": obj,
            "object_top3": [[obj_names[i], round(float(obj_scores[i]), 4)] for i in ranked[:3]],
            "parts": {class_names[k]: round(c / label.size, 4)
                      for k, c in zip(*np.unique(label, return_counts=True)) if k != BACKGROUND},
        })
        del specific, part_obj
    return labels, best, records


def label_map(scores, class_names, bg_on, objects, min_score):
    """Argmax over parts -> uint16 label map; BACKGROUND where background wins, where the best part
    scores below min_score, or (with --objects) for parts of other objects."""
    n = len(class_names)
    part_scores = scores[:n].copy()
    if objects:
        keep = np.array([c.split("'s")[0].strip() in objects for c in class_names])
        part_scores[~keep] = -1.0
    labels = part_scores.argmax(0).astype(np.uint16)
    best = part_scores.max(0)
    background = best < min_score
    if bg_on:
        background |= scores[n] > best
    labels[background] = BACKGROUND
    return labels, best


def summarise(labels, best, class_names):
    total = labels.size
    found = []
    for idx, count in zip(*np.unique(labels, return_counts=True)):
        if idx == BACKGROUND:
            continue
        found.append({
            "part": class_names[idx],
            "pixel_fraction": round(count / total, 5),
            "mean_score": round(float(best[labels == idx].mean()), 4),
        })
    return sorted(found, key=lambda d: -d["pixel_fraction"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="image file, or a directory (its first image is used)")
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["paco"])
    ap.add_argument("--weights", help="checkpoint (.pth) to load instead of the model's default; "
                                      "needs exactly one --models entry")
    ap.add_argument("--vocab", nargs="+", help="zero-shot vocabulary, e.g. \"mug's handle\" \"kettle's spout\"")
    ap.add_argument("--vocab-file", help="zero-shot vocabulary file, one \"<object>'s <part>\" per line")
    ap.add_argument("--output-dir", default=str(REPO / "results"))
    ap.add_argument("--scene-name", help="results sub-folder (default: derived from the image path)")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="only predict parts of these objects, e.g. mug (PACO), bottle (VOC), glass (ADE)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="pixels whose best part score is below this are left unlabelled")
    ap.add_argument("--obj-masks", nargs="+", metavar="PNG",
                    help="one binary mask per object (e.g. GraspClutter6D mask_visib/<frame>_*.png, or "
                         "SAM masks); runs the model per object crop, conditioned on the predicted object")
    ap.add_argument("--crop-size", type=int, default=384,
                    help="--obj-masks: each object crop is resized to this square (training crops are 384)")
    ap.add_argument("--crop-margin", type=float, default=0.15,
                    help="--obj-masks: context around the object, as a fraction of its size per side")
    ap.add_argument("--min-mask-area", type=int, default=100, help="--obj-masks: skip smaller masks (px)")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay opacity")
    ap.add_argument("--min-area", type=int, default=None,
                    help="skip drawing parts smaller than this many pixels (default: 0.05%% of image)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("PartCATSeg needs a GPU (its CLIP text encoder is hard-wired to .cuda()).")
    if args.weights and len(args.models) != 1:
        sys.exit("--weights needs exactly one --models entry (the config that matches the checkpoint)")
    vocab = read_vocab(args.vocab, args.vocab_file)
    test_dataset = register_vocab(vocab) if vocab else None
    if vocab:
        print(f"zero-shot vocabulary ({len(vocab)}): {vocab}")

    image_path = resolve_image(args.image)
    scene = args.scene_name or scene_name_for(image_path)
    out_dir = Path(args.output_dir) / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    stem = image_path.stem
    min_area = args.min_area if args.min_area is not None else int(0.0005 * image.shape[0] * image.shape[1])
    masks = load_masks(args.obj_masks, image.shape[:2]) if args.obj_masks else None
    # distinct output names per mode, so runs with different options don't overwrite each other
    # (mask runs are tagged with the masks' directory, e.g. _mask_visib for GT vs _sam3_masks)
    tag = "".join([f"_{Path(args.obj_masks[0]).resolve().parent.name}" if masks else "",
                   f"_ms{args.min_score:g}" if args.min_score else ""])
    print(f"image : {image_path}  ({image.shape[1]}x{image.shape[0]})\noutput: {out_dir}")
    if masks:
        print(f"masks : {len(masks)} object masks")
    Image.fromarray(image).save(out_dir / f"{stem}_input.png")

    panels = [image]
    for name in args.models:
        print(f"\n[{name}] loading model ...")
        cfg, model = load_model(name, args.weights, test_dataset)
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        class_names = list(meta.stuff_classes)

        objects = None
        if args.objects:
            model_objects = {c.split("'s")[0].strip() for c in class_names}
            objects = set(args.objects) & model_objects
            if not objects:
                print(f"[{name}] none of {args.objects} are in this model's vocabulary {sorted(model_objects)}; skipping")
                del model
                continue

        print(f"[{name}] running inference ...")
        prefix = out_dir / f"{stem}_{name}{tag}"
        object_records = None
        if masks:
            labels, best, object_records = label_map_per_object(
                cfg, model, image, masks, class_names, objects, args.min_score,
                args.crop_size, args.crop_margin, args.min_mask_area)
            vis = Visualizer(image, metadata=meta, instance_mode=ColorMode.IMAGE)
            for (_, mask), rec in zip(sorted(masks, key=lambda pm: -pm[1].sum()), object_records):
                if "object" in rec:
                    vis.draw_binary_mask(mask, text=rec["object"], alpha=args.alpha)
            Image.fromarray(vis.output.get_image()).save(f"{prefix}_objects.png")
            for rec in object_records:
                if "object" in rec:
                    print(f"[{name}]   {Path(rec['mask']).name:<20} obj_id {rec['gc6d_obj_id']}  -> "
                          + ", ".join(f"{o} {s:.2f}" for o, s in rec["object_top3"]))
        else:
            scores = predict(cfg, model, image)
            labels, best = label_map(scores, class_names, model.sem_seg_head.predictor.bg_on,
                                     objects, args.min_score)
            del scores

        vis = Visualizer(image, metadata=meta, instance_mode=ColorMode.IMAGE)
        overlay = vis.draw_sem_seg(labels, area_threshold=min_area, alpha=args.alpha).get_image()
        Image.fromarray(overlay).save(f"{prefix}_parts.png")
        Image.fromarray(labels).save(f"{prefix}_labels.png")  # uint16, 65535 = none

        parts = summarise(labels, best, class_names)
        with open(f"{prefix}_parts.json", "w") as f:
            json.dump({"image": str(image_path), "model": name, "checkpoint": cfg.MODEL.WEIGHTS,
                       "custom_vocab": bool(vocab), "objects_filter": sorted(objects or []),
                       "min_score": args.min_score, "label_png_background_value": int(BACKGROUND),
                       "obj_masks": object_records is not None, "crop_size": args.crop_size,
                       "crop_margin": args.crop_margin, "objects": object_records,
                       "class_names": class_names, "parts": parts}, f, indent=2)
        print(f"[{name}] top parts: " + ", ".join(f"{p['part']} {100 * p['pixel_fraction']:.1f}%" for p in parts[:8]))
        panels.append(overlay)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(panels) > 1:
        Image.fromarray(np.concatenate(panels, axis=1)).save(out_dir / f"{stem}{tag}_comparison.png")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
