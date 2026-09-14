#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
job="$1"; device="$2"
case "$job" in
 single_upstream) bundle=single; mode=upstream;;
 single_uniform) bundle=single; mode=uniform;;
 four_uniform) bundle=four; mode=uniform;;
 four_adaptive) bundle=four; mode=adaptive;;
 *) exit 2;;
esac
test -s /data/code/ouyanghao/oyh/acs_archive_20260915/FINALIZED.json
mkdir -p records/tracking_diagnostics_20260915/locks
exec 9>"records/tracking_diagnostics_20260915/locks/${job}.lock"
flock -n 9 || exit 4
free_kb=$(df -Pk /data | awk 'NR==2 {print $4}')
if ((free_kb < 9437184)); then echo 'DISK_GATE: 8 GiB reserve plus 1 GiB output'; exit 3; fi
logdir=records/tracking_diagnostics_20260915
refdir=runs/tracking_diagnostics_20260915_refs
output="runs/tracking_diagnostic_${job}_seed0"
test ! -e "$output"
bash scripts/run_isaaclab.sh scripts/train_tracking_diagnostic.py --bundle "$refdir/${bundle}.npz" --output "runs/tracking_diagnostic_smoke_${job}" --mode "$mode" --num_envs 32 --smoke-steps 300 --headless --device "$device" > "$logdir/smoke_${job}.log" 2>&1
test -s "runs/tracking_diagnostic_smoke_${job}/SMOKE_COMPLETE.json"
bash scripts/run_isaaclab.sh scripts/train_tracking_diagnostic.py --bundle "$refdir/${bundle}.npz" --output "$output" --mode "$mode" --num_envs 512 --iterations 1000 --seed 0 --headless --device "$device" > "$logdir/train_${job}.log" 2>&1
test -s "$output/TRAINING_COMPLETE.json"
test -s "$output/model_999.pt"
while IFS= read -r reference; do
 id=$(basename "$reference" _actor0.npz)
 bash scripts/run_isaaclab.sh scripts/evaluate_g1_tracking.py --checkpoint "$output/model_999.pt" --motion_file "$reference" --output "runs/tracking_diagnostic_eval_${job}_${id}" --num_envs 32 --seed 123 --headless --device "$device" > "$logdir/eval_${job}_${id}.log" 2>&1
done < "$refdir/evaluate_training_paths.txt"
for reference in runs/kungfu_test4_tracking_refs/sample_*_actor0.npz; do
 id=$(basename "$reference" _actor0.npz)
 bash scripts/run_isaaclab.sh scripts/evaluate_g1_tracking.py --checkpoint "$output/model_999.pt" --motion_file "$reference" --output "runs/tracking_diagnostic_dev_${job}_${id}" --num_envs 32 --seed 123 --held-out --headless --device "$device" > "$logdir/dev_${job}_${id}.log" 2>&1
done
echo "TRACKING_DIAGNOSTIC_JOB_COMPLETE $job"
