#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
export PATH="$root/.venv-pbhc-training/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4
export WANDB_MODE=disabled WANDB_SILENT=true MPLBACKEND=Agg
export LD_LIBRARY_PATH="$root/.venv-pbhc-training/lib:$root/third_party/isaacgym/python/isaacgym/_bindings/linux-x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$root/third_party/PBHC:$root/third_party/PBHC/humanoidverse${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$root/cache/torch-extensions-pbhc"
export XDG_CACHE_HOME="$root/cache/xdg-pbhc"
unset CUDA_VISIBLE_DEVICES
cd "$root/third_party/PBHC"
exec "$root/.venv-pbhc-training/bin/python" -u "$@"
