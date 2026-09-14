#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="$1"
device="$2"
mode="$3"
while kill -0 "$parent" 2>/dev/null; do sleep 10; done
checkpoint=runs/multireference_kungfu20_seed0/model_999.pt
test -s runs/multireference_kungfu20_seed0/TRAINING_COMPLETE.json
test -s "$checkpoint"
if [[ "$mode" == single ]]; then
  for split in train20 test4; do
    for reference in runs/kungfu_${split}_tracking_refs/sample_*_actor0.npz; do
      id="$(basename "$reference" _actor0.npz)"
      output="runs/g1_tracking_eval_multiref_${split}_${id}_primed"
      bash scripts/run_isaaclab.sh scripts/evaluate_g1_tracking.py --checkpoint "$checkpoint" --motion_file "$reference" --output "$output" --num_envs 32 --headless --device "$device" > "records/paired_v1/eval_multiref_${split}_${id}.log" 2>&1
      test -s "$output/metrics.json"
    done
  done
elif [[ "$mode" == dual ]]; then
  test -s runs/actor_reaction_kinematic_contact_tracking_refs/manifest.json
  for reference in runs/actor_reaction_kinematic_contact_tracking_refs/sample_*_actor0.npz; do
    id="$(basename "$reference" _actor0.npz)"
    other="${reference%_actor0.npz}_actor1.npz"
    output="runs/dual_g1_contact_${id}_multiref999"
    bash scripts/run_isaaclab.sh scripts/evaluate_dual_g1_tracking.py --reference-a "$reference" --reference-b "$other" --checkpoint "$checkpoint" --output "$output" --num-envs 4 --headless --device "$device" > "records/paired_v1/eval_dual_contact_${id}.log" 2>&1
    test -s "$output/metrics.json"
  done
else
  printf '%s\n' 'Expected single or dual'; exit 2
fi
printf '%s\n' "MULTIREFERENCE_${mode}_EVALUATION_COMPLETE"
