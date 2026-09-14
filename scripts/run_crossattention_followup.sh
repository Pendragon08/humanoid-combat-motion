#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 10; done
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_timotion.sh scripts/train_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --output runs/actor_reaction_crossattention_seed0 --history-only --cross-attention --steps 1000 --seed 0 > records/paired_v1/actor_reaction_crossattention_train.log 2>&1
test -s runs/actor_reaction_crossattention_seed0/TRAINING_COMPLETE.json
bash scripts/run_timotion.sh scripts/infer_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --adapter runs/actor_reaction_crossattention_seed0/best_adapter.pt --output runs/actor_reaction_crossattention_test_seed123 --limit 20 --seed 123 > records/paired_v1/actor_reaction_crossattention_inference.log 2>&1
test -s runs/actor_reaction_crossattention_test_seed123/evaluation.json
printf '%s\n' CROSSATTENTION_EXPERIMENT_COMPLETE
