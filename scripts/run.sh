#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=egl
export XDG_CACHE_HOME="$PWD/cache/xdg" MPLCONFIGDIR="$PWD/cache/matplotlib"
export TMPDIR="$PWD/cache/tmp" PYTHONPYCACHEPREFIX="$PWD/cache/pycache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PWD/.venv/bin/python" -u "$@"
