#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
rec="$root/records/pbhc_official_reproduction_20260917"
mkdir -p "$rec" "$root/cache/downloads"
test "$(git -C third_party/PBHC rev-parse HEAD)" = ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0
test -z "$(git -C third_party/PBHC diff --name-only)"
if [ ! -f cache/downloads/IsaacGym_Preview_4.tar.gz ]; then
  curl -fL --retry 2 --connect-timeout 20 --max-time 600 https://developer.nvidia.com/isaac-gym-preview-4 -o cache/downloads/IsaacGym_Preview_4.tar.gz.partial
  tar -tzf cache/downloads/IsaacGym_Preview_4.tar.gz.partial >/dev/null
  mv cache/downloads/IsaacGym_Preview_4.tar.gz.partial cache/downloads/IsaacGym_Preview_4.tar.gz
fi
sha256sum cache/downloads/IsaacGym_Preview_4.tar.gz > "$rec/isaacgym_archive.sha256"
if [ ! -d third_party/isaacgym ]; then
  tar -xzf cache/downloads/IsaacGym_Preview_4.tar.gz -C third_party
fi
test -f third_party/isaacgym/python/setup.py
if [ ! -x .venv-pbhc-training/bin/python ]; then
  /data/code/ouyanghao/miniconda3/bin/conda create -y -p "$root/.venv-pbhc-training" python=3.8.20 pip
fi
export PIP_CACHE_DIR="$root/cache/pip-pbhc-training"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4
.venv-pbhc-training/bin/python - <<'PY'
import json, subprocess, sys
spec=json.load(open('env-pbhc-training.json'))
for phase in spec['pip_phases']:
    subprocess.run([sys.executable,'-m','pip','install','-c','configs/pbhc_training_constraints.txt']+phase,check=True)
PY
.venv-pbhc-training/bin/python -m pip freeze > "$rec/packages.txt"
printf 'PBHC_ENV_INSTALLED_NOT_YET_VALIDATED\n'
