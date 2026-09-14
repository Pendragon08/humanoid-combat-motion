#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Isaac/Vulkan and CUDA enumerate GPUs differently under this variable.
# Select the physical GPU with AppLauncher's --device cuda:N instead.
unset CUDA_VISIBLE_DEVICES
export WANDB_MODE=disabled WANDB_SILENT=true
export XDG_CACHE_HOME="$PWD/cache/xdg" MPLCONFIGDIR="$PWD/cache/matplotlib"
export TMPDIR="$PWD/cache/tmp" PIP_CACHE_DIR="$PWD/cache/pip-isaac"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec "$PWD/third_party/isaacsim-4.5.0/python.sh" -u "$@"
