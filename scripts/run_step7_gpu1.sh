#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_step7_utility_job.sh real 1 cuda:1
bash scripts/run_step7_utility_job.sh real 2 cuda:1
test -s runs/step7_bundles/protocol.json
bash scripts/run_step7_utility_job.sh generic 2 cuda:1
bash scripts/run_step7_utility_job.sh contact 2 cuda:1
echo STEP7_GPU1_COMPLETE
