#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
for seed in 0 1 2; do
  run="runs/timotion_adapter_fixedrng_seed${seed}"
  bash scripts/run_timotion.sh scripts/train_timotion_adapter.py --checkpoint data/official/timotion_weights.npz --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --output "$run" --seed "$seed" --steps 1000 --batch-size 4 --frames 120 > "records/paired_v1/adapter_fixedrng_seed${seed}.log" 2>&1
  bash scripts/run_timotion.sh scripts/train_timotion_adapter.py --checkpoint data/official/timotion_weights.npz --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --output "${run}_validation" --evaluate-adapter "$run/best_adapter.pt" --seed "$seed" --batch-size 4 --frames 120 > "records/paired_v1/adapter_fixedrng_seed${seed}_validation.log" 2>&1
done
bash scripts/run_timotion.sh scripts/infer_timotion.py --checkpoint data/official/timotion_weights.npz --prompts configs/combat_prompts_24.json --output runs/timotion_generic_24_seed2026 --seed 2026 --frames 150 > records/paired_v1/generic24_generation.log 2>&1
bash scripts/run_timotion.sh scripts/infer_timotion.py --checkpoint data/official/timotion_weights.npz --adapter runs/timotion_adapter_fixedrng_seed0/best_adapter.pt --conditions configs/combat_conditions_24.json --prompts configs/combat_prompts_24.json --output runs/timotion_adapted_24_seed2026 --seed 2026 --frames 150 > records/paired_v1/adapted24_generation.log 2>&1
for arm in generic adapted; do
  bash scripts/run_timotion.sh scripts/stabilize_generated_identity.py --input "runs/timotion_${arm}_24_seed2026" --output "runs/timotion_${arm}_24_seed2026_identity" > "records/paired_v1/${arm}24_identity.log" 2>&1
  bash scripts/run_retarget.sh scripts/retarget_paired_positions.py --input "runs/timotion_${arm}_24_seed2026_identity" --output "runs/timotion_${arm}_24_seed2026_g1ik" > "records/paired_v1/${arm}24_g1ik.log" 2>&1
done
printf '%s\n' 'ADAPTER_EXPERIMENT_QUEUE_COMPLETE'
