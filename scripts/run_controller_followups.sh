#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="$1"
while kill -0 "$parent" 2>/dev/null; do sleep 10; done
checkpoint=logs/rsl_rl/g1_flat/2026-09-14_12-31-18_seed199_expanded1024/model_3097.pt
test -s "$checkpoint"
bash scripts/run_isaaclab.sh scripts/evaluate_g1_tracking.py --checkpoint "$checkpoint" --motion_file data/tracking_seed/199_train_120.npz --output runs/g1_tracking_eval_iter3097_primed --num_envs 32 --headless --device cuda:0 > records/paired_v1/tracking_eval_iter3097.log 2>&1
test -s runs/g1_tracking_eval_iter3097_primed/metrics.json
bash scripts/run_isaaclab.sh scripts/convert_paired_tracking_references.py --input runs/kungfu_test6_qpos --output runs/kungfu_test4_tracking_refs --headless --device cuda:0 > records/paired_v1/kungfu_test4_conversion.log 2>&1
test -s runs/kungfu_test4_tracking_refs/manifest.json
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timotion.sh scripts/train_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --output runs/actor_reaction_history_seed0 --history-only --seed 0 --steps 1000 > records/paired_v1/actor_reaction_history_training.log 2>&1
test -s runs/actor_reaction_history_seed0/TRAINING_COMPLETE.json
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timotion.sh scripts/infer_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --adapter runs/actor_reaction_history_seed0/best_adapter.pt --output runs/actor_reaction_history_test_seed123 --limit 20 --seed 123 > records/paired_v1/actor_reaction_history_inference.log 2>&1
test -s runs/actor_reaction_history_test_seed123/evaluation.json
printf '%s\n' CONTROLLER_FOLLOWUPS_COMPLETE
