#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
record=records/tracking_diagnostics_20260915
mkdir -p "$record"
exec 9>"$record/orchestrator.lock"
flock -n 9 || exit 2
while [[ ! -s /data/code/ouyanghao/oyh/acs_archive_20260915/FINALIZED.json ]]; do
 if [[ -f /data/code/ouyanghao/oyh/acs_archive_20260915/ARCHIVE_FAILED.json || -f /data/code/ouyanghao/oyh/acs_archive_20260915/FINALIZER_FAILED.json ]]; then
  echo 'ARCHIVE_GATE_FAILED: no experiments launched'; exit 3
 fi
 sleep 30
done
bash scripts/run.sh scripts/prepare_tracking_diagnostics.py > "$record/prepare.log" 2>&1
bash scripts/run.sh -m unittest tests.test_diagnostic_sampling -v > "$record/sampling_tests.log" 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$record/gpu_before.csv"
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>=500 {busy=1} END {exit !busy}'; then
 echo 'GPU_GATE: another workload is active'; exit 4
fi
(
 bash scripts/run_tracking_diagnostic_job.sh single_upstream cuda:0
 bash scripts/run_tracking_diagnostic_job.sh four_uniform cuda:0
) > "$record/gpu0_queue.log" 2>&1 &
gpu0=$!
(
 bash scripts/run_tracking_diagnostic_job.sh single_uniform cuda:1
 bash scripts/run_tracking_diagnostic_job.sh four_adaptive cuda:1
) > "$record/gpu1_queue.log" 2>&1 &
gpu1=$!
printf '{"gpu0_queue_pid":%s,"gpu1_queue_pid":%s,"status":"launched_after_verified_archive_cleanup"}\n' "$gpu0" "$gpu1" > "$record/launch.json"
echo "DIAGNOSTIC_QUEUES_LAUNCHED $gpu0 $gpu1"
status=0
wait "$gpu0" || status=1
wait "$gpu1" || status=1
if ((status)); then echo DIAGNOSTIC_QUEUE_FAILED; exit 5; fi
echo TRACKING_DIAGNOSTICS_FOUR_JOBS_COMPLETE
