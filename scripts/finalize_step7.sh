#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for parent in "$@"; do
  while kill -0 "$parent" 2>/dev/null; do sleep 20; done
done
bash scripts/run.sh scripts/summarize_step7.py --html reports/experiment_progress_20260914.html
bash scripts/run.sh -c 'import json; r=json.load(open("records/step7/summary.json")); assert r["pilot_runs_complete"], "A queue ended before all nine runs completed"'
bash scripts/run.sh scripts/render_step7_utility.py --output runs/step7_utility_media > records/step7/render.log 2>&1
cp runs/step7_utility_media/step7_utility.mp4 reports/media/
cp runs/step7_utility_media/step7_utility_preview.png reports/media/
cp runs/step7_utility_media/provenance.json reports/media/step7_utility_provenance.json
bash scripts/run.sh scripts/summarize_step7.py --html reports/experiment_progress_20260914.html
bash scripts/run.sh scripts/collect_step7_records.py
echo STEP7_PILOT_ARTIFACTS_COMPLETE
