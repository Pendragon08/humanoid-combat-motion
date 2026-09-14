#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
arm="$1"; seed="$2"; device="$3"
case "$arm" in real) bundle=data/tracking_seed/kungfu_train20_bundle.npz ;; generic|contact) bundle="runs/step7_bundles/${arm}.npz" ;; *) exit 2 ;; esac
output="runs/step7_utility_${arm}_seed${seed}"
test -s "$bundle"
test ! -e "$output"
free_kb=$(df -Pk /data | awk 'NR==2 {print $4}')
if ((free_kb < 8388608)); then echo 'DISK_RESERVE_GATE: need 8 GiB'; exit 3; fi
bash scripts/run_isaaclab.sh scripts/train_multireference_tracking.py --bundle "$bundle" --output "$output" --iterations 1000 --num_envs 512 --seed "$seed" --headless --device "$device" > "records/step7/train_${arm}_${seed}.log" 2>&1
test -s "$output/TRAINING_COMPLETE.json"
test -s "$output/model_999.pt"
for reference in runs/kungfu_test4_tracking_refs/sample_*_actor0.npz; do
  id=$(basename "$reference" _actor0.npz)
  bash scripts/run_isaaclab.sh scripts/evaluate_g1_tracking.py --checkpoint "$output/model_999.pt" --motion_file "$reference" --output "runs/step7_eval_${arm}_seed${seed}_${id}" --num_envs 32 --seed 123 --held-out --headless --device "$device" > "records/step7/eval_${arm}_${seed}_${id}.log" 2>&1
  test -s "runs/step7_eval_${arm}_seed${seed}_${id}/metrics.json"
done
printf '%s\n' "STEP7_JOB_COMPLETE $arm $seed"
