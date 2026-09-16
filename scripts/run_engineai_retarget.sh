#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
# Existing overlays share Python 3.10. Retarget packages take priority; the pilot
# overlay supplies its already-installed OSQP 1.0.5 without modifying either env.
export PYTHONPATH="$PWD:$PWD/.venv-retarget/lib/python3.10/site-packages:$PWD/.venv/lib/python3.10/site-packages"
exec "$PWD/.venv-retarget/bin/python" -u "$@"
