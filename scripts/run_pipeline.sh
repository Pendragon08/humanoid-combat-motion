#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed="${1:?seed required}"
test -f audit/tdmpc2_archive/cleanup_verified.json
bash scripts/run.sh -m unittest discover -s tests -v
bash scripts/run.sh scripts/train.py --seed "$seed" --steps 3000 --run "pilot_seed${seed}"
bash scripts/run.sh scripts/evaluate.py --run "pilot_seed${seed}"
if [ "$seed" = 0 ]; then
    bash scripts/run.sh scripts/render_results.py --run pilot_seed0
    bash scripts/run.sh scripts/export_reference.py --input runs/pilot_seed0/motions.npz --case 0 --arm 3 --output runs/pilot_seed0/example_tracking_reference.npz
fi
bash scripts/run.sh -c "import json,time; from pathlib import Path; Path('runs/pipeline_seed${seed}.complete.json').write_text(json.dumps({'seed':${seed},'finished_unix':time.time(),'status':'PASS'}))"
