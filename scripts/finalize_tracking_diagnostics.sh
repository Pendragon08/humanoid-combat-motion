#!/usr/bin/env bash
# One bounded finalizer for the already-authorized four experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
record=records/tracking_diagnostics_20260915
mkdir -p "$record"
exec 9>"$record/finalizer.lock"
flock -n 9 || exit 2
while true; do
 python3 scripts/summarize_tracking_diagnostics.py
 if python3 -c 'import json; d=json.load(open("records/tracking_diagnostics_20260915/summary.json")); raise SystemExit(0 if d["complete_jobs"]==4 else 1)'; then break; fi
 if grep -Eq 'DIAGNOSTIC_QUEUE_FAILED|ARCHIVE_GATE_FAILED|GPU_GATE' "$record/orchestrator.log"; then
  echo FINALIZER_QUEUE_FAILURE; exit 3
 fi
 sleep 30
done
python3 scripts/summarize_tracking_diagnostics.py --html reports/experiment_progress_20260914.html
python3 scripts/collect_tracking_diagnostic_records.py
for comparison in interface sampling; do
 output="runs/tracking_diagnostic_video_${comparison}"
 if [[ ! -e "$output" ]]; then
  bash scripts/run.sh scripts/render_tracking_diagnostics.py --comparison "$comparison" --output "$output"
 fi
 test -s "$output/provenance.json"
 test -s "$output/comparison.mp4"
 mkdir -p reports/media
 cp "$output/comparison.mp4" "reports/media/tracking_diagnostic_${comparison}.mp4"
 cp "$output/preview.png" "reports/media/tracking_diagnostic_${comparison}.png"
done
python3 scripts/summarize_tracking_diagnostics.py --html reports/experiment_progress_20260914.html
python3 scripts/collect_tracking_diagnostic_records.py
date -u +%FT%TZ > "$record/FINAL_REPORT_READY"
echo TRACKING_DIAGNOSTICS_FINAL_REPORT_READY
