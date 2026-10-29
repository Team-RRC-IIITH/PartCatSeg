# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PartCATSeg (CVPR 2025): open-vocabulary **part** segmentation built on CAT-Seg's cost-aggregation
design, as a Detectron2 project. Upstream: `github.com/kaist-cvml/part-catseg`.

## Commands

There is no build step, linter, or test suite for the project code (`open_clip/tests` belongs to the
vendored library). Everything runs through the one entry point, `train_net.py`, a Detectron2
`DefaultTrainer` launcher. Run it from the repo root: dataset roots and weight paths in the configs are
relative.

```sh
# train
python train_net.py --num-gpus 4 --config-file configs/zero_shot/partcatseg_voc.yaml

# evaluate a checkpoint
python train_net.py --num-gpus 4 --config-file configs/zero_shot/partcatseg_voc.yaml \
    --eval-only MODEL.WEIGHTS ./weights/partcatseg_voc.pth

# any config key can be overridden as trailing KEY VALUE pairs, e.g. a quick 1-GPU smoke run
python train_net.py --num-gpus 1 --config-file configs/zero_shot/partcatseg_voc.yaml \
    SOLVER.IMS_PER_BATCH 1 SOLVER.MAX_ITER 10 OUTPUT_DIR output/smoke
```

Configs: `configs/zero_shot/{partcatseg,catseg}_{voc,ade,partimagenet,partimagenet_ood}.yaml`, which
inherit from `configs/base_{partcatseg,catseg}_config.yaml` through `_BASE_`. `setup()` calls
`cfg.set_new_allowed(True)`, so **a mistyped config key is silently accepted, not rejected**.

Dataset preprocessing (one-off, see README for downloads):
`baselines/data/datasets/mask_cls_collect.py` (PascalPart116 label counts) and
`datasets/partimagenet_preprocess.py` (needs `datasets/LOC_synset_mapping.txt`).

## Required files on disk

- `pretrain_weights/model_final_base.pth` (CAT-Seg weights). Every PartCATSeg config sets it as
  `MODEL.WEIGHTS`, and `DINOv2Backbone.__init__` also reads it directly to pull the `backbone.dinov2.*`
  tensors.
- Datasets under `datasets/` (`PascalPart116`, `ADE20KPart234`, `PartImageNet`, `PartImageNet_OOD`).
  The root can be overridden with `$DETECTRON2_DATASETS`.
- Internet access on first run: the DINOv2 backbone is fetched with
  `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, and the CLIP ViT-B/16 weights are
  downloaded by `baselines/third_party/clip.py`.

## Environment (pixi)

The env is defined in `pixi.toml` / `pixi.lock` and lives in `.pixi/envs/default`.
**Always `source ./pixi_setup.sh` before any `pixi` command.** pixi takes its cache location from
env vars, not from the manifest, and its default `~/.cache/rattler` is on `/`, which is ~100% full
and shared by every user of this server. `[activation.env]` additionally points the runtime caches
(`TORCH_HOME`, `HF_HOME`, `CLIP_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR`) into the project, so
`pixi run` / `pixi shell` never write to `~`.

```sh
source ./pixi_setup.sh && pixi install          # create / sync the env
pixi run python train_net.py --num-gpus 1 --config-file configs/zero_shot/partcatseg_voc.yaml ...
```

Never `pip install` into this env. Add packages with `pixi add` / `pixi add --pypi` so the lock
stays authoritative. torch 2.2.2+cu121 is a **conda** package (pytorch channel), and detectron2 is
compiled against it from a pinned git commit (`no-build-isolation`, using the env's `cuda-nvcc`
12.1; there is no system nvcc). A PyPI torch arriving by any route would break detectron2's
`_C` extension.

Deliberate deviations from the README, each explained in `pixi.toml`:
- Pillow is 9.4 (`<10`), not 9.5.
- `mmcv` is the ops-free version, not `mmcv-full`.
- mmsegmentation is omitted.
- **`open_clip_torch` is not installed.** With it installed, `import open_clip` imports the
  vendored `transformers/` and crashes, so ViT-H/ViT-G CLIP don't work (same as upstream).

`requirements.txt` is stale (it pins torch 1.12.1).

**Run `pixi run fetch-dinov2` once after `pixi install`.** `DINOv2Backbone` loads
`torch.hub.load('facebookresearch/dinov2', ...)` without a pinned version, so it fetches DINOv2's
latest `main`. Since 2025-06-11 that code uses `float | None` (Python >= 3.10) and fails with
`TypeError: unsupported operand type(s) for |`. `fetch_dinov2.py` fills torch.hub's cache dir
(`$TORCH_HOME/hub/facebookresearch_dinov2_main`) with commit `85a2460`, which was `main` when
PartCATSeg was released. torch.hub then reuses that copy ("Using cache found in ...").
**Never pass `force_reload=True`**: it would re-download the incompatible code.

## Single-image inference

`infer_image.py <image or dir> [--models paco|voc|ade] [--weights X.pth] [--vocab ...|--vocab-file F]
[--objects ...] [--min-score ...]` writes `results/<scene>/`. The default model is `paco`
(`output/partcatseg_paco_tabletop/model_final.pth`); `--weights` loads any checkpoint for a single model.
`--vocab`/`--vocab-file` give a zero-shot vocabulary of `"<object>'s <part>"` names. They are
registered as metadata-only and set as `DATASETS.TEST`, which is where the predictor builds its test
text embeddings, so nothing is retrained. Outputs:
- `*_parts.png`: overlay with part names;
- `*_labels.png`: uint16 label map, 65535 = no part;
- `*_parts.json`;
- `*_comparison.png`: the input next to each model's overlay.

It reproduces the inference branch of `PartCATSeg.forward` with one difference: there is **no
ground-truth object conditioning**. The benchmark masks every part channel that doesn't belong to
the GT object and scores only pixels inside the object, so the model never learned to reject
background. On whole images, the background channel (loss weight 0.05) almost never wins and
nearly every pixel gets a part label. Use `--objects` and/or `--min-score` to limit this.
On GC6D scene 000000 it takes ~5 GB of GPU and ~40 s once the weights are cached.

`--obj-masks <png ...>` (one binary mask per object, e.g. GraspClutter6D
`mask_visib/<frame>_*.png`) fixes this. It runs the model once per object: a square crop around the
mask (`--crop-margin` 0.15) resized to 384² (`--crop-size`; the training crop size), conditioned on the
object class with the highest mean object score inside the mask. The object class comes from the
object channels of the part+object head. Only pixels inside a mask are labelled, and `*_objects.png`
shows the predicted object per mask. On GC6D 000000 frame 1 (16 GT masks) it takes ~15 s, and its
boundaries follow the objects, unlike the whole-image mode, which yields ~80×45 px blobs because
CLIP sees the 16:9 image squashed to a 24×24 patch grid. Object classes there are right for the
mugs, cans and most boxes, but wrong for some boxes (book / carton / mug) and are inconsistent
across instances that share an obj_id. Output names carry a mode tag
(`_<mask dir name>`, e.g. `_mask_visib` / `_sam3_masks`, and `_ms<min-score>`) so runs don't overwrite each other.

`run_part_seg.sh` wraps both mask sources: set `MODE` (gt | sam3 | both), `SCENE_DIR`, `FRAME`
(or `IMAGE`), `WEIGHTS` and `EXTRA_ARGS` at the top, then run `bash run_part_seg.sh`.

SAM 3 masks: `export_sam3_masks.py <image>` writes `results/<scene>/sam3_masks/<stem>_sam3_*.png`.
SAM 3 needs transformers 5, so it runs with **SegObjectMaterial's** env, called directly, not via
`pixi run`. That repo's `pixi.toml` still sets `HF_HOME`/`TORCH_HOME` to the old
`/scratch2/soumo_roy/...` path, so `pixi run` would try to re-download the gated weights:
`HF_HOME=/scratch2/ennovatex.Soumo/.cache/huggingface HF_HUB_OFFLINE=1 /scratch2/ennovatex.Soumo/SegObjectMaterial/.pixi/envs/default/bin/python export_sam3_masks.py <img>`.
The script drops this repo from `sys.path` because the vendored `transformers/` would shadow 5.x.
On GC6D 000000 frame 1, prompt `item` returns 16 objects against 16 GT, with near-GT masks. The
per-object classes are about as good as with GT masks but flip on a few objects (pitcher
cup↔bottle, small bottle → cup). Object scores are often 0.06–0.2, so the object choice, not the
masks, is now the weak link.

Fixed 2026-10-28: the model's logits cover the *padded* input (ImageList pads to a multiple of 32).
`sem_seg_postprocess` crops by input-pixel size on the low-res logit grid, i.e. not at all, so the
padded rows were stretched over the image, shifting predictions by up to ~4 % vertically.
`infer_image.forward` now crops the logits proportionally before upsampling. Upstream's evaluator
has the same issue and was left untouched.

`GraspClutter6D` now lives at `/scratch2/ennovatex.Rath/dataset/GraspClutter6D-2` (another user's
tree: read only).

`baselines/third_party/clip.py:_download` was changed to honour `CLIP_CACHE_DIR`. Its default
(`~/.cache/clip`) is unchanged.

## PACO tabletop dataset (added for GraspClutter6D part segmentation)

The goal is part segmentation of tabletop objects (GraspClutter6D scenes). Neither VOC's nor
ADE's vocabulary covers those objects, so the model is trained on **PACO-LVIS restricted to 39
tabletop objects / 197 object-part classes**. Examples: bottle, box, can, cup, mug, jar, bowl,
plate, knife, scissors, drill.

| stage | location |
|---|---|
| raw annotations | `/scratch2/ennovatex.Soumo/part_datasets/PACO/annotations/paco_lvis_v1_{train,val,test}.json` |
| images | `/scratch2/ennovatex.Soumo/part_datasets/coco/{train,val}2017/` — only the ~57.6k PACO images, fetched by `PACO/download_images.sh` |
| converted | `/scratch2/ennovatex.Soumo/part_datasets/PACO_tabletop/` (by `datasets/paco_tabletop_preprocess.py`) |
| registration | `baselines/data/datasets/register_paco_tabletop.py`; registers nothing if `classes.json` is absent |
| config | `configs/zero_shot/partcatseg_paco_tabletop.yaml` |
| train + test | `bash train_paco.sh [KEY VALUE ...]` (env: `OUT`, `NUM_GPUS`, `FORCE`); resumes, then scores `model_final.pth` on PACO test |

About the images:
- The script fetches them one at a time, 48 in parallel. A single connection to
  `images.cocodataset.org` runs at ~67 KB/s from this server, which would make the 20 GB COCO zips
  a multi-day download.
- The script is resumable: re-run it to fill gaps. All 57,643 present as of 2026-10-28.
- The data root was renamed from `/scratch2/soumo_roy` to `/scratch2/ennovatex.Soumo` (2026-10-28).
  The converter bakes **absolute** paths into `PACO_tabletop/*.json`, so after any move re-run it.

`DETECTRON2_DATASETS` is set to `/scratch2/ennovatex.Soumo/part_datasets` in `pixi.toml`, which the VOC
and ADE configs need too.

The tabletop object list is the `TABLETOP` dict in the converter. Change the list there and
re-run the converter (~30 s, CPU only).

Design decisions:
- **No zero-shot split:** every class is trained on. The evaluator reports `mIoU-base` over all
  classes and skips the `-unbase` metrics.
- Splits are PACO's own. Train on `train`, validate on `val` (`*_val_obj_condition`); `test` is
  the original LVIS val set.
- The object map is **pre-rendered** (`annotations_detectron2_obj/`) rather than drawn by the
  mapper from object polygons. Every part pixel overrides it with its own parent's class.
  Otherwise overlapping objects (a mug inside a bowl) leave mug parts inside the "bowl" region,
  and conditioning on "bowl" supervises and scores mug parts. This was caught by a check and
  fixed.
- Object display names are cleaned for CLIP ("drinking glass", "computer mouse", …). **No object
  name may be a substring of another.** Upstream's object-conditioned inference selects parts
  with `obj_class in class_name`, so "pen" would also admit "pencil" parts; PACO's pen is
  therefore named "ink pen". The converter asserts this.
- Training starts from `weights/partcatseg_ade.pth` (fine-tune). The paper instead starts from
  the CAT-Seg base checkpoint, which is not downloaded here.
- In COCO images the tabletop objects are usually small; parts cover ~6 % of pixels per image.
- **One 4090 cannot hold upstream's batch of 4** (22 GB at iter 0, OOM): upstream runs it as
  4 GPUs x 1 image. The PACO config therefore uses `IMS_PER_BATCH 1` + `SOLVER.ACCUM_STEPS 4`
  (`AccumTrainer` in `train_net.py`), which gives the same effective batch and LR schedule. Measured:
  16.1 GB peak, 1.23 s/step (about 6.8 h for 20k steps), and val eval about 11 min per pass (4.3 GB).

## Architecture

Registration happens at import time. `train_net.py` imports `baselines`, which registers:
- the meta-architectures `CATSeg`, `PartCATSeg`, `ObjCATSeg` and `ObjPartCATSeg` (`baselines/*.py`);
- the heads (`baselines/modeling/heads/`) and backbones such as `DINOv2Backbone` and the Swin variants;
- every dataset split, through the `register_*.py` modules in `baselines/data/datasets/`.

The config picks components by name: `MODEL.META_ARCHITECTURE`, `MODEL.BACKBONE.NAME`,
`MODEL.SEM_SEG_HEAD.NAME`. Only `CATSeg` and `PartCATSeg` have evaluators wired up in
`train_net.py:evaluator_classes`. The `Obj*` variants are present but commented out there.

**Data flow (`baselines/partcatseg.py: PartCATSeg.forward`)**
1. The image is normalised two ways. The CLIP copy is resized to 384² (ViT-B/16) and run through
   `clip_model.encode_image(dense=True)`, which gives the dense CLIP features. The pixel copy goes to
   `DINOv2Backbone`, which resizes it to 448² and produces ViT-S/14 patch tokens, interpolated to
   `res2/res3/res4` at 96/48/24 (the "appearance guidance").
2. `PartCATSegHead`, then `PartCATSegPredictor`, then `Aggregator`
   (`modeling/transformer/part_cat_seg_*.py`) build image–text cost volumes against two sets of
   text embeddings:
   - **part + object** text: the generic part names plus the object names, e.g. "wheel", "car";
   - **specific part** text: the original `"object's part"` class names.
   The Aggregator has three parallel layer stacks. `layers` and `layers_object` serve the
   part/object pathway, and `layers_specific_part` serves the object-specific parts. Two embedding
   convs, `conv1` and `conv2`, sit alongside them. In `PartCATSeg.__init__`, `layers_object` and
   `layers_specific_part` are **initialised as copies of `layers`**, and `conv2` as a copy of `conv1`.
   After that, `layers_object` is frozen.
3. The training loss is BCE on the specific-part output, plus BCE on the part and object outputs,
   plus a Jensen–Shannon divergence term. The JSD term compares the object cost with the
   specific-part cost scattered onto the objects. A learned "non-object" text embedding is appended
   when `BG_ON`, with a loss weight of 0.05.

**The class vocabulary is derived from strings.** Dataset metadata exposes `stuff_classes` in the
form `"object's part"`, plus `obj_classes`. `init_metadata`, which exists in both the meta-arch and
the predictor, splits on `'s` to build the part list, the object list and the index maps
(`text_to_part_map` and similar). New dataset class names must follow the `"<obj>'s <part>"` format.
Base/novel splits are defined by the `BASE_CLASS_NAMES` lists in each `register_*.py`.

**Evaluation is object-conditioned.** Test splits are the `*_obj_condition` variants, which yield one
record per object instance. At inference the model reads the ground-truth object class from
`instances` and sets every part channel not belonging to that object to -1. The code only reads
`batched_inputs[0]`, so test batch size must stay 1.
`baselines/evaluation/partcatseg_evaluation.py` reports `mIoU-base`, `mIoU-unbase` and `h-IoU`
(their harmonic mean). For **PartImageNet_OOD, use `mIoU-unbase`**: its val set has no base classes.

**Dataset mappers** are selected by `INPUT.DATASET_MAPPER_NAME`:
- `obj_part_semantic` for Pascal and ADE;
- `partimagenet`;
- `partimagenet_ood`.

These produce `obj_part_sem_seg` (part labels), `sem_seg` (object labels) and `instances`. The ignore
value is 255, except on PartImageNet, where it is 65535.

## Vendored code

- `baselines/third_party/clip.py` is a modified OpenAI CLIP. It supports `encode_image(dense=True)`
  and prompt-depth arguments. ViT-B/16 loads through it.
- `open_clip/` is the full open_clip repo, with the package under `open_clip/src/open_clip`. It is
  only used for `CLIP_PRETRAINED` set to `ViT-H` or `ViT-G`. It is not installed, so
  `import open_clip` resolves to this directory as an empty namespace package (see Environment).
- `transformers/` is a cut-down copy of HuggingFace `transformers` 4.29.0.dev0. Its `models/`
  holds only auto, clip, clipseg, partclipseg and vit. The project code never imports it, but it
  shadows any installed `transformers` for Python run from the repo root. Because it is
  incomplete, a failed import raises `RuntimeError` instead of `ImportError`, which defeats
  optional-import guards such as open_clip's.
