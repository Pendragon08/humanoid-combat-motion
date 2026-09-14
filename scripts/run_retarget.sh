#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
export PYTHONPATH="$PWD:$PWD/third_party/GMR${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$PWD/cache/xdg" TMPDIR="$PWD/cache/tmp"
exec "$PWD/.venv-retarget/bin/python" -u "$@"
