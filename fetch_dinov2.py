"""Pin the DINOv2 code that DINOv2Backbone loads through torch.hub.

`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` fetches DINOv2's *current* main
branch. Since 2025-06-11 (#528) that code uses `float | None` annotations (Python >= 3.10), so it
no longer imports under this project's Python 3.8. torch.hub reuses an existing cache directory
instead of re-downloading, so this script fills that directory with the commit that was `main`
when PartCATSeg was released (2025-04-15). No project code is changed.

    pixi run fetch-dinov2          # once per checkout; --force to replace an existing copy
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "85a24602099d397264d5b30461ad7f3bfd726ca1"  # main from 2024-10-26 to 2025-05-19
URL = f"https://github.com/facebookresearch/dinov2/archive/{COMMIT}.zip"


def main():
    hub_dir = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub"
    target = hub_dir / "facebookresearch_dinov2_main"  # the cache dir torch.hub looks for
    stamp = target / ".pinned_commit"
    if stamp.is_file() and stamp.read_text().strip() == COMMIT and "--force" not in sys.argv:
        print(f"already pinned: {target} @ {COMMIT[:12]}")
        return

    print(f"downloading DINOv2 @ {COMMIT[:12]} ...")
    with urllib.request.urlopen(URL, timeout=300) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = zf.namelist()[0].split("/")[0]
        tmp = hub_dir / f".dinov2_{COMMIT[:12]}"
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)

    shutil.rmtree(target, ignore_errors=True)
    hub_dir.mkdir(parents=True, exist_ok=True)
    (tmp / root).rename(target)
    shutil.rmtree(tmp, ignore_errors=True)
    stamp.write_text(COMMIT + "\n")

    bad = [p for p in target.rglob("*.py") if "| None" in p.read_text(errors="ignore")]
    if bad:
        sys.exit(f"pinned code still uses Python>=3.10 syntax: {bad[:3]}")
    print(f"pinned: {target} @ {COMMIT[:12]}")


if __name__ == "__main__":
    main()
