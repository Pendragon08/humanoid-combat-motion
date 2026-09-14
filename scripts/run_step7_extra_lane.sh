#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
arm="$1"; device="$2"
test -s runs/step7_bundles/protocol.json
for seed in 1 2; do bash scripts/run_step7_utility_job.sh "$arm" "$seed" "$device"; done
echo "STEP7_EXTRA_LANE_COMPLETE $arm"
