#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint='logs/rsl_rl/g1_flat/2026-09-14_11-36-31_seed199_extended/model_1098.pt'
# This frozen controller covers one old seed only. These are diagnostics, not
# approved critic labels or retrained-policy synthetic-data utility comparisons.
for arm in generic adapted; do
  references="runs/timotion_${arm}_24_seed2026_tracking_refs"
  bash scripts/run_isaaclab.sh scripts/convert_paired_tracking_references.py --input "runs/timotion_${arm}_24_seed2026_g1ik" --output "$references" --headless --device cuda:1 > "records/paired_v1/${arm}24_reference_conversion.log" 2>&1
  for number in $(seq 0 23); do
    sample=$(printf 'sample_%03d' "$number")
    output="runs/dual_g1_${arm}_${sample}_iter1098"
    bash scripts/run_isaaclab.sh scripts/evaluate_dual_g1_tracking.py --reference-a "$references/${sample}_actor0.npz" --reference-b "$references/${sample}_actor1.npz" --checkpoint "$checkpoint" --output "$output" --headless --device cuda:1 > "records/paired_v1/dual_${arm}_${sample}.log" 2>&1
    printf '%s\n' "DUAL_CASE_COMPLETE $arm $sample"
  done
done
printf '%s\n' 'DUAL_DIAGNOSTIC_QUEUE_COMPLETE'
