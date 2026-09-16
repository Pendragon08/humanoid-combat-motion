#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl WANDB_MODE=disabled WANDB_SILENT=true
export PYTHONPATH="$PWD:$PWD/third_party/PBHC:$PWD/third_party/PBHC/humanoidverse:$PWD/third_party/PBHC/humanoidverse/isaac_utils${PYTHONPATH:+:$PYTHONPATH}"
exec "$PWD/.venv-pbhc-inference/bin/python" -u "$@"
