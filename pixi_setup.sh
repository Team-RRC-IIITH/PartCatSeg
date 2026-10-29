# Source this BEFORE any `pixi` command:   source ./pixi_setup.sh
#
# pixi reads its cache location from the environment, not from pixi.toml, so without this it
# downloads into ~/.cache/rattler on `/`, which is ~100% full and shared by every user on this box.
# The runtime caches (torch hub, CLIP, HuggingFace) are redirected separately by [activation.env]
# in pixi.toml, so `pixi run` / `pixi shell` stay inside the project too.

_PCS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"

export PIXI_CACHE_DIR="$_PCS_ROOT/.pixi_cache/rattler"
export RATTLER_CACHE_DIR="$PIXI_CACHE_DIR"
export UV_CACHE_DIR="$_PCS_ROOT/.pixi_cache/uv"
export PIP_CACHE_DIR="$_PCS_ROOT/.pixi_cache/pip"
export XDG_CACHE_HOME="$_PCS_ROOT/.cache"
export TMPDIR="$_PCS_ROOT/.tmp"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

mkdir -p "$PIXI_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
unset _PCS_ROOT
