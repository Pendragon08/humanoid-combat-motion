#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
export WANDB_MODE=disabled WANDB_SILENT=true
export XDG_CACHE_HOME="$PWD/cache/engineai-official/xdg" MPLCONFIGDIR="$PWD/cache/engineai-official/matplotlib"
export TMPDIR="$PWD/cache/engineai-official/tmp" PIP_CACHE_DIR="$PWD/cache/engineai-official/pip"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
lab="$PWD/third_party/IsaacLab-engineai-official"
export PYTHONPATH="$lab/source/isaaclab:$lab/source/isaaclab_assets:$lab/source/isaaclab_tasks:$lab/source/isaaclab_rl:$PWD/third_party/engineai_rl_lab/source/engineai_rl_lab:$PWD/.venv-engineai-official/lib/python3.10/site-packages:$PWD"
unset PYTHONEXE
# Use the simulator's matching interpreter/stdlib; the isolated venv supplies
# packages only. Mixing conda's Python build string with Kit's stdlib is invalid.
exec "$PWD/third_party/isaacsim-4.5.0/python.sh" -u scripts/engineai_official_entry.py "$@"
