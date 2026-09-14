#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/cache/xdg" MPLCONFIGDIR="$PWD/cache/matplotlib" TMPDIR="$PWD/cache/tmp"
export PYTHONPYCACHEPREFIX="$PWD/cache/pycache-timotion"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export PYTHONPATH="$PWD:$PWD/third_party/TIMotion:$PWD/third_party/CLIP${PYTHONPATH:+:$PYTHONPATH}"
exec "$PWD/.venv-timotion/bin/python" -u "$@"
